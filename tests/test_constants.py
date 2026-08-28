import torch

from yiwm.constants import (
    BINARY_TO_KING_WEN, HEX_ADJ, HEX_CUO, HEX_HU, HEX_LOWER_TRIGRAM,
    HEX_UPPER_TRIGRAM, HEX_ZONG, KING_WEN_TO_BINARY, TRIGRAM_INDEX,
)


def _bin_of_kw(n):  # King Wen number -> internal index
    return KING_WEN_TO_BINARY[n - 1].item()


def test_king_wen_anchors():
    assert _bin_of_kw(1) == 63          # 乾為天 = 111111
    assert _bin_of_kw(2) == 0           # 坤為地 = 000000
    assert _bin_of_kw(3) == 17          # 水雷屯  = 震下坎上 -> 100 010
    assert _bin_of_kw(29) == 18         # 坎為水  = 010 010
    assert _bin_of_kw(30) == 45         # 離為火  = 101 101
    assert _bin_of_kw(63) == 21         # 既濟   = 離下坎上 -> 101 010
    assert _bin_of_kw(64) == 42         # 未濟   = 坎下離上 -> 010 101


def test_king_wen_is_permutation():
    assert sorted(KING_WEN_TO_BINARY.tolist()) == list(range(64))
    round_trip = BINARY_TO_KING_WEN[KING_WEN_TO_BINARY]
    assert torch.equal(round_trip, torch.arange(64))


def test_adjacency_is_hamming1():
    assert torch.equal(HEX_ADJ.sum(1), torch.full((64,), 6.0))
    assert torch.equal(HEX_ADJ, HEX_ADJ.t())
    assert HEX_ADJ.diagonal().sum() == 0


def test_cuo_zong_involutions():
    assert torch.equal(HEX_CUO[HEX_CUO], torch.arange(64))
    assert torch.equal(HEX_ZONG[HEX_ZONG], torch.arange(64))
    # 乾/坤/坎/離/頤/大過/中孚/小過 are self-綜
    for k in (63, 0, 18, 45):
        assert HEX_ZONG[k].item() == k


def test_hu_of_qian_kun():
    assert HEX_HU[63].item() == 63       # 乾 nuclear = 乾
    assert HEX_HU[0].item() == 0         # 坤 nuclear = 坤


def test_trigram_split_matches_index():
    # 乾為天: lower & upper both 乾
    assert HEX_LOWER_TRIGRAM[63].item() == TRIGRAM_INDEX["qian"]
    assert HEX_UPPER_TRIGRAM[63].item() == TRIGRAM_INDEX["qian"]
    # 水雷屯 (idx 17): lower 震, upper 坎
    assert HEX_LOWER_TRIGRAM[17].item() == TRIGRAM_INDEX["zhen"]
    assert HEX_UPPER_TRIGRAM[17].item() == TRIGRAM_INDEX["kan"]
