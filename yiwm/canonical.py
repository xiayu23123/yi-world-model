"""Fill `data/yao_seed.json`'s text fields:
  * `canonical_text` from `CANONICAL` -- public-domain 爻辞. Only 乾/坤 ship
    (transcribing 372 from memory is error-prone); paste the rest from a 周易
    source into `CANONICAL` or pass `extra=`.
  * `modern_text` from `MODERN_P0` -- the 12 乾/坤 P0 anchors (the two poles of
    the force space). Pass `modern_extra=` for P1/P2 rows you write by hand.

Keys: hexagram name as in `constants.KING_WEN_NAMES` (乾, 坤, 屯, ...); yao name
as emitted by `seed._yao_name` (初九 ... 上九 / 初六 ... 上六).

Run: python -m yiwm.canonical data/yao_seed.json
"""

from __future__ import annotations

import json
from pathlib import Path

CANONICAL: dict[str, dict[str, str]] = {
    "乾": {
        "初九": "潜龙勿用",
        "九二": "见龙在田，利见大人",
        "九三": "君子终日乾乾，夕惕若厉，无咎",
        "九四": "或跃在渊，无咎",
        "九五": "飞龙在天，利见大人",
        "上九": "亢龙有悔",
    },
    "坤": {
        "初六": "履霜，坚冰至",
        "六二": "直方大，不习无不利",
        "六三": "含章可贞，或从王事，无成有终",
        "六四": "括囊，无咎无誉",
        "六五": "黄裳，元吉",
        "上六": "龙战于野，其血玄黄",
    },
}


# P0 anchor `modern_text` for 乾/坤 -- the two poles of the force space.
# ~100 chars, no 卦/爻/阴/阳 vocab, structure mapped (timing + 老爻 + 应).
MODERN_P0: dict[str, dict[str, str]] = {
    "乾": {
        "初九": "一位资深算法专家离职创业，目前仅有核心代码库和一台服务器。他拒绝媒体采访，也不参加行业峰会，而是潜入目标客户公司做临时顾问以贴近真实需求。资金只够八个月，他决定不设办公室，全员远程。",
        "九二": "初创产品在种子用户里口碑良好，日活连续三周稳步增长。创始人决定接受天使轮，并着手组建销售团队。市场上已有巨头布局，但该产品在垂直细分领域的解决效率领先对手一倍。",
        "九三": "项目完成 B 轮，团队扩至八十人，核心业务进入稳定期。主要对手正发起价格战。创始人决定短期不跟风降价，转而优化内部供应链，把两成研发资源投向风险较低的 B 端服务探索。",
        "九四": "公司排行业第二，份额与第一只差五个百分点。董事会要求要么激进扩张夺份额，要么收缩战线等待被收购。创始人同时测试两条路：三个试点城市推扩张，同时保留核心产研团队做安全垫。",
        "九五": "企业已是行业标准制定者，股价创新高，产品毛利率七成半。监管警告其处理供应链里的垄断嫌疑。他主动放慢扩张，把年度利润的三成投入公益与人才培养，以巩固长期生态位。",
        "上九": "创始人已退居二线，仍握董事会否决权。对新任 CEO 的数字化转型方案，他凭旧经验强烈反对，内部派系斗争公开化。他意识到再坚持公司会失去新一代市场，于是把否决权让渡给年轻合伙人。",
    },
    "坤": {
        "初六": "宏观数据连续两月低于预期，股市仍在涨。一家跨国企业的财务负责人注意到大宗商品价格异常波动，决定暂缓一亿美元的原料采购计划，改为按周分批小量采购，观察政策风向。",
        "六二": "一位年轻律师在顶级律所头三年默默深耕某一细分法案，从未独立出庭，但她整理的法律意见书在所内获合伙人高度认可。合伙人外出期间，她主动接手一起跨部门案件，积累了关键人脉。",
        "六三": "地方一家中型制造企业承接头部国企的外包订单，营收稳定但议价权极低。老板靠内部技术改进把良品率提到九成八，却选择不申请专利也不对外展示，作为隐性资产维持现有订单的稳定交付。",
        "六四": "大型集团的审计总监察觉某子公司财务异常，涉事方是创始家族成员。正式上报可能被边缘化，隐瞒则担职业风险。他先以内部风险评估名义留存书面记录，同时低调接触外部猎头留后路。",
        "六五": "她主持的慈善基金会规模已过百亿，声誉极高。她不以救火者形象示人，而是专注搭建长线教育支持系统。有政商人士建议她借影响力涉足商业，她仍坚守捐赠章程，把重心放在标准化的项目评估上。",
        "上六": "两大股东团围绕一块未开发地块的产权爆发对峙，地方调解无效，已升级到诉讼。作为中立第三方，他判断这将是漫长的消耗战，遂放弃单纯追求公正的裁断，转而推动双方进入仲裁，把系统性损失压到最低。",
    },
}


# The `synth._action` rule is a toy heuristic and misreads 乾/坤 (it has no model
# of 柔顺 / 刚健). For the hand-authored P0 anchors, these labels from the design
# tables override the derived `action` so prose and label agree.
P0_ACTION: dict[str, dict[str, str]] = {
    "乾": {"初九": "dai", "九二": "jin", "九三": "shou", "九四": "bian", "九五": "shou", "上九": "tui"},
    "坤": {"初六": "dai", "六二": "jin", "六三": "shou", "六四": "tui", "六五": "shou", "上六": "bian"},
}


def import_canonical(seed_path: str, extra: dict | None = None,
                     modern_extra: dict | None = None) -> dict:
    """Fill `canonical_text` (from CANONICAL + `extra`) and `modern_text`
    (from MODERN_P0 + `modern_extra`) in place."""
    ctab: dict[str, dict[str, str]] = {k: dict(v) for k, v in CANONICAL.items()}
    for h, m in (extra or {}).items():
        ctab.setdefault(h, {}).update(m)
    mtab: dict[str, dict[str, str]] = {k: dict(v) for k, v in MODERN_P0.items()}
    for h, m in (modern_extra or {}).items():
        mtab.setdefault(h, {}).update(m)

    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    filled = modern = relabel = 0
    for row in seed:
        h, y = row["hex_name"], row["yao_name"]
        if ctab.get(h, {}).get(y):
            row["canonical_text"] = ctab[h][y]
            filled += 1
        if mtab.get(h, {}).get(y):
            row["modern_text"] = mtab[h][y]
            row["verified"] = True
            modern += 1
            act = P0_ACTION.get(h, {}).get(y)
            if act and act != row["action"]:
                row["action_derived"] = row["action"]
                row["action"] = act
                relabel += 1
    Path(seed_path).write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"canonical_filled": filled, "modern_filled": modern, "p0_relabelled": relabel,
            "total": len(seed),
            "hexagrams_covered": sum(1 for h in ctab if any(ctab[h].values()))}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/yao_seed.json"
    print(import_canonical(path))
