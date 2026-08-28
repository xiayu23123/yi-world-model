import torch

from yiwm.textenc import get_text_encoder, hash_bag


def test_hash_encoder_shape_and_determinism():
    fn, dim = get_text_encoder("hash")
    assert dim == 256
    a = fn(["处于「生」之时。初爻阳，势强，当位有应。", "another"])
    b = fn(["处于「生」之时。初爻阳，势强，当位有应。", "another"])
    assert a.shape == (2, 256)
    assert torch.equal(a, b)                       # deterministic
    assert a.abs().amax() <= 1.0 + 1e-6            # row-normalised


def test_hash_has_no_synonym_generalisation():
    # documents the known limitation the sentence-transformer path fixes
    v = hash_bag(["融资紧缺", "筹资困难"])
    cos = torch.cosine_similarity(v[0:1], v[1:2]).item()
    assert cos < 0.3                               # near-orthogonal despite being synonyms


def test_unknown_encoder_errors():
    try:
        get_text_encoder("bogus")
        assert False
    except ValueError:
        pass


def test_frozen_sentence_transformer_if_available():
    import pytest
    fn = dim = None
    for name in ("minilm-ml", "minilm"):
        try:
            fn, dim = get_text_encoder(name)
            break
        except RuntimeError:
            continue
    if fn is None:
        pytest.skip("no sentence-transformer model available offline")
    assert dim == 384
    emb = fn(["hello world", "goodbye"])
    assert emb.shape == (2, 384)
    assert not emb.requires_grad                   # frozen: no grad path
    assert not emb.is_inference()                  # clone() stripped inference-mode


def test_multilingual_clusters_synonyms_if_available():
    import pytest
    import torch.nn.functional as F
    try:
        fn, _ = get_text_encoder("minilm-ml")
    except RuntimeError:
        pytest.skip("minilm-ml model not available offline")
    e = fn(["蛰伏之时，静待时机", "潜藏之时，不宜妄动", "天气晴朗适合出游"])
    syn = F.cosine_similarity(e[0:1], e[1:2]).item()
    unrel = F.cosine_similarity(e[0:1], e[2:3]).item()
    assert syn > unrel + 0.2                       # zero-shot synonym clustering
