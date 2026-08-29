import torch

from yiwm.model import YiWorldModel
from yiwm.structured_input import QUESTIONS, choices_to_yao, structured_infer


def test_choices_to_yao_shape_and_sign():
    y = choices_to_yao({"phase": "5", "polarity": "1", "resource": "1", "domain": "3"})
    assert y.shape == (6,)
    assert (y > 0).all()                       # polarity 1 (阳) -> all positive
    assert y.argmax().item() == 4              # phase 5 -> 五爻 strongest

    yn = choices_to_yao({"phase": "1", "polarity": "2", "resource": "3", "domain": "1"})
    assert (yn < 0).all()                       # polarity 2 (阴) -> all negative
    assert yn.abs().argmax().item() == 0        # phase 1 -> 初爻 strongest
    assert yn.abs().max() < choices_to_yao(
        {"phase": "1", "polarity": "2", "resource": "1", "domain": "1"}).abs().max()  # 紧缺 scales down


def test_all_option_combos_produce_valid_yao():
    for p in QUESTIONS["phase"]["options"]:
        for pol in QUESTIONS["polarity"]["options"]:
            for r in QUESTIONS["resource"]["options"]:
                y = choices_to_yao({"phase": p, "polarity": pol, "resource": r, "domain": "1"})
                assert torch.isfinite(y).all() and y.abs().max() <= 1.0


def test_structured_infer_end_to_end():
    m = YiWorldModel(obs_dim=35)               # encoder is bypassed
    out = structured_infer(
        {"phase": "4", "polarity": "1", "resource": "2", "domain": "1"}, m
    )
    # uniform polarity -> 本卦 乾(1) (all yang)
    assert out["ben_hex"][0] == 1
    # 动爻 = the phase line itself; 之卦 = 乾 with 九四 flipped -> 风天小畜 (KW 9)
    assert out["moving_lines"] == [0, 0, 0, 1, 0, 0]
    assert out["zhi_hex"][0] == 9
    assert set(out["action"]) == {"进", "退", "守", "变", "待"}


def test_phase_drives_moving_line():
    m = YiWorldModel(obs_dim=35)
    for ph in range(1, 7):
        out = structured_infer(
            {"phase": str(ph), "polarity": "2", "resource": "1", "domain": "2"}, m
        )
        assert out["ben_hex"][0] == 2                     # all yin -> 坤
        assert out["moving_lines"][ph - 1] == 1 and sum(out["moving_lines"]) == 1
