"""
基于多视图群体匹配与三支决策的学生发展导航智能体（实验脚本）

数据集：data.csv（字段：编号, 专业, 高考, 综测, GPA, 最高英语, 竞赛, 科研成果, 实习, 志愿时长, 学生职务, 性格, 家庭, 最终出路）

严格落地：
1) 多视图特征构建与加权（含目标驱动的动态视图权重）
2) 群体相似性挖掘：数值欧式距离 + 类别杰卡德系数
3) 三支决策：正域/边界域/负域

运行：
  python 0312.py --csv data.csv --target "考研"
输出：
  控制台打印论文式“计算过程 + Top-5群体编号 + 动态路径 + 三支决策报告”
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 0. 配置：字段/视图/权重（严格按题设）
# -----------------------------

ID_COL = "编号"
TARGET_COL = "最终出路"

NUM_COLS_ACADEMIC = ["高考", "GPA", "综测", "最高英语"]
NUM_W_ACADEMIC = {"高考": 0.3, "GPA": 0.4, "综测": 0.2, "最高英语": 0.1}

NUM_COLS_ACHIEVEMENT = ["竞赛", "科研成果"]
NUM_W_ACHIEVEMENT = {"竞赛": 0.6, "科研成果": 0.4}

NUM_COLS_PRACTICE = ["实习", "志愿时长", "学生职务"]
NUM_W_PRACTICE = {"实习": 0.5, "志愿时长": 0.3, "学生职务": 0.2}

CAT_COLS_BACKGROUND = ["性格", "家庭", "专业"]
# 背景视图内部：性格0.5、家庭0.5（专业不计入W4内部，仅用于杰卡德类别相似度的“匹配约束/加成”）
CAT_W_BACKGROUND = {"性格": 0.5, "家庭": 0.5}

# 目标驱动的动态视图权重（W1..W4）
DYNAMIC_VIEW_WEIGHTS: Dict[str, Dict[str, float]] = {
    "考研": {"W1": 0.6, "W2": 0.1, "W3": 0.2, "W4": 0.1},
    "保研": {"W1": 0.5, "W2": 0.3, "W3": 0.1, "W4": 0.1},
    "考公": {"W1": 0.2, "W2": 0.1, "W3": 0.5, "W4": 0.2},
    "留学": {"W1": 0.4, "W2": 0.2, "W3": 0.2, "W4": 0.2},
    "就业": {"W1": 0.3, "W2": 0.1, "W3": 0.5, "W4": 0.1},
}


# -----------------------------
# 1. 工具函数：配置与读CSV（自动编码）、清洗、标准化
# -----------------------------


def _load_dotenv_if_exists() -> None:
    """
    轻量级 .env 加载（不依赖第三方库）：
    - 支持 KEY=VALUE 形式
    - 忽略以 # 开头的注释行与空行
    - 不覆盖已经存在于环境中的变量
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:  # noqa: BLE001
        return
    env_path = os.path.join(base_dir, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, value = s.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # 配置文件加载失败不应阻断主流程
        return

def _try_read_csv_with_encodings(csv_path: str, encodings: Iterable[str]) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            # encoding_errors: Python 3.10+ / pandas 1.5+ 支持，遇到脏字节也能读入（用于“实验可复现”优先）
            return pd.read_csv(csv_path, encoding=enc, encoding_errors="strict")
        except Exception as e:  # noqa: BLE001 - 需要尝试多编码
            last_err = e
    raise RuntimeError(
        f"无法读取CSV：{csv_path}。已尝试编码={list(encodings)}；最后错误={type(last_err).__name__}: {last_err}"
    )


def _sniff_file_type_and_encoding(csv_path: str) -> Tuple[str, Optional[str]]:
    """
    返回：(file_type, encoding_hint)
    - file_type: "csv" or "excel"
    - encoding_hint: 可能的编码提示；若无法判断则None
    """
    raw = open(csv_path, "rb").read(4096)

    # Excel xlsx/xlsm 通常是ZIP容器（PK头）
    if raw.startswith(b"PK\x03\x04"):
        return "excel", None

    # UTF-16/UTF-32 常见特征：大量空字节
    if raw.count(b"\x00") > len(raw) * 0.2:
        # 进一步用BOM判断
        if raw.startswith(b"\xff\xfe\x00\x00") or raw.startswith(b"\x00\x00\xfe\xff"):
            return "csv", "utf-32"
        if raw.startswith(b"\xff\xfe"):
            return "csv", "utf-16le"
        if raw.startswith(b"\xfe\xff"):
            return "csv", "utf-16be"
        return "csv", "utf-16"

    # UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        return "csv", "utf-8-sig"

    # 尝试用第三方库猜测（若存在则用；不存在则跳过）
    try:
        import chardet  # type: ignore

        det = chardet.detect(open(csv_path, "rb").read())
        enc = det.get("encoding")
        return "csv", enc
    except Exception:
        return "csv", None


def load_dataset(csv_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到文件：{csv_path}")

    file_type, enc_hint = _sniff_file_type_and_encoding(csv_path)

    if file_type == "excel":
        # 允许用户把Excel直接命名成data.csv的情况
        df = pd.read_excel(csv_path)
    else:
        # 1) 若用户指定编码，则优先使用（并允许replace以保证能跑完实验）
        if encoding:
            try:
                df = pd.read_csv(csv_path, encoding=encoding, encoding_errors="strict")
            except Exception:
                df = pd.read_csv(csv_path, encoding=encoding, encoding_errors="replace")
        else:
            # 2) 启发式 + 常见编码兜底
            encodings = []
            if enc_hint:
                encodings.append(enc_hint)
            encodings.extend(["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "utf-16", "utf-16le", "utf-16be"])

            # 先严格读（避免无声乱码），失败再replace保证可运行
            try:
                df = _try_read_csv_with_encodings(csv_path, encodings=encodings)
            except RuntimeError:
                last_err: Optional[Exception] = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc, encoding_errors="replace")
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                else:
                    raise RuntimeError(
                        f"无法读取CSV：{csv_path}。已尝试(含replace)编码={encodings}；最后错误={type(last_err).__name__}: {last_err}"
                    )

    # 统一列名去空格
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        ID_COL,
        "专业",
        *NUM_COLS_ACADEMIC,
        *NUM_COLS_ACHIEVEMENT,
        *NUM_COLS_PRACTICE,
        "性格",
        "家庭",
        TARGET_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要字段：{missing}；实际字段={list(df.columns)}")

    # 数值列尽量转为float（保留NaN用于缺失判定）
    for col in (NUM_COLS_ACADEMIC + NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 类别列统一为字符串（缺失保留NaN）
    for col in ["专业", "性格", "家庭", TARGET_COL]:
        df[col] = df[col].astype("string")

    # 编号强制为字符串，方便展示
    df[ID_COL] = df[ID_COL].astype("string")
    return df


@dataclass(frozen=True)
class StudentCase:
    专业: str
    高考: float
    综测: float
    GPA: float
    最高英语: float
    竞赛: float
    科研成果: float
    实习: float
    志愿时长: float
    学生职务: float
    性格: str
    家庭: str
    目标: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "专业": self.专业,
            "高考": self.高考,
            "综测": self.综测,
            "GPA": self.GPA,
            "最高英语": self.最高英语,
            "竞赛": self.竞赛,
            "科研成果": self.科研成果,
            "实习": self.实习,
            "志愿时长": self.志愿时长,
            "学生职务": self.学生职务,
            "性格": self.性格,
            "家庭": self.家庭,
            "目标": self.目标,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StudentCase":
        def _get(k: str, default: Any) -> Any:
            return d.get(k, default)

        return StudentCase(
            专业=str(_get("专业", "数科")),
            高考=float(_get("高考", 578)),
            综测=float(_get("综测", 77)),
            GPA=float(_get("GPA", 3.25)),
            最高英语=float(_get("最高英语", 468)),
            竞赛=float(_get("竞赛", 0)),
            科研成果=float(_get("科研成果", 0)),
            实习=float(_get("实习", 0)),
            志愿时长=float(_get("志愿时长", 38)),
            学生职务=float(_get("学生职务", 0)),
            性格=str(_get("性格", "偏内向")),
            家庭=str(_get("家庭", "一般")),
            目标=str(_get("目标", "考研")).strip(),
        )


def minmax_params(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    params: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            params[c] = (0.0, 0.0)
        else:
            params[c] = (float(s.min()), float(s.max()))
    return params


def minmax_scale(x: float, c: str, params: Dict[str, Tuple[float, float]]) -> float:
    mn, mx = params[c]
    if not np.isfinite(x):
        return float("nan")
    if mx == mn:
        return 0.0
    return (float(x) - mn) / (mx - mn)


# -----------------------------
# 2. 相似度：数值欧式距离 + 类别杰卡德
# -----------------------------

def weighted_euclidean_distance(
    x: Dict[str, float],
    y: Dict[str, float],
    weights: Dict[str, float],
    mm_params: Dict[str, Tuple[float, float]],
    *,
    missing_policy: str = "penalize",
) -> Tuple[float, Dict[str, float]]:
    """
    在 min-max 标准化空间计算加权欧式距离：
      d = sqrt( sum_i w_i * (x_i' - y_i')^2 )
    返回：距离d，以及每个字段的加权平方差贡献（便于展示计算过程）

    缺失值策略（用于严格实验口径）：
    - penalize: 若任一侧缺失，则按最大差异(1.0)计入该维度贡献：w_i*(1)^2
      解释：缺失意味着无法确认相似，保守地降低相似度，避免“缺失导致虚高”。
    - ignore:   若任一侧缺失，则跳过该维度（可能导致相似度偏乐观，不建议用于论文主结果）。
    """
    contrib: Dict[str, float] = {}
    s = 0.0
    for k, w in weights.items():
        xs = minmax_scale(x[k], k, mm_params)
        ys = minmax_scale(y[k], k, mm_params)
        if (not np.isfinite(xs)) or (not np.isfinite(ys)):
            if missing_policy == "ignore":
                contrib[k] = float("nan")
                continue
            # 保守惩罚：按最大差异计入
            term = float(w) * (1.0**2)
            contrib[k] = term
            s += term
            continue
        term = float(w) * (xs - ys) ** 2
        contrib[k] = term
        s += term
    return math.sqrt(s), contrib


def euclid_to_similarity(d: float) -> float:
    """
    将距离映射为(0,1]相似度：sim = 1 / (1 + d)
    说明：题设只规定“欧式距离”，未规定融合到[0,1]的形式；此处采用常用单调映射，便于与杰卡德融合。
    """
    if not np.isfinite(d):
        return 0.0
    return 1.0 / (1.0 + float(d))


def jaccard_similarity(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    A = {t for t in a_tokens if t is not None and str(t).strip() != ""}
    B = {t for t in b_tokens if t is not None and str(t).strip() != ""}
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def tokenize_category(v: Optional[str]) -> List[str]:
    """
    将类别值转为token集合。若字段里包含“/、,、;、|、空格”等分隔符，视作多标签。
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):  # type: ignore[arg-type]
        return []
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    for sep in ["/", "、", ",", "，", ";", "；", "|", " "]:
        s = s.replace(sep, "|")
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts if parts else [str(v).strip()]


# -----------------------------
# 3. 多视图融合：按动态W1..W4加权求总相似度
# -----------------------------

@dataclass
class SimilarityBreakdown:
    id: str
    target: str
    sim_w1: float
    sim_w2: float
    sim_w3: float
    sim_w4: float
    sim_total: float
    # 过程细节（用于论文式展示）
    w1_detail: Dict[str, float]
    w2_detail: Dict[str, float]
    w3_detail: Dict[str, float]
    w4_jaccard_components: Dict[str, float]
    missing_key_info: List[str]


def compute_similarity(
    df: pd.DataFrame,
    case: StudentCase,
    view_weights: Dict[str, float],
    key_fields_for_missing: List[str],
) -> List[SimilarityBreakdown]:
    # min-max参数按全体样本估计（实验常见设定）
    mm_params = minmax_params(df, NUM_COLS_ACADEMIC + NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE)

    out: List[SimilarityBreakdown] = []

    case_num = {
        "高考": case.高考,
        "综测": case.综测,
        "GPA": case.GPA,
        "最高英语": case.最高英语,
        "竞赛": case.竞赛,
        "科研成果": case.科研成果,
        "实习": case.实习,
        "志愿时长": case.志愿时长,
        "学生职务": case.学生职务,
    }
    case_cat = {"专业": case.专业, "性格": case.性格, "家庭": case.家庭}

    for _, r in df.iterrows():
        sid = str(r[ID_COL])
        target = str(r[TARGET_COL])

        # 缺失信息判定（用于三支决策边界域触发）
        missing_fields: List[str] = []
        for f in key_fields_for_missing:
            if f in (NUM_COLS_ACADEMIC + NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE):
                if not np.isfinite(r[f]):
                    missing_fields.append(f)
            else:
                v = r.get(f, None)
                if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                    missing_fields.append(f)

        # W1/W2/W3：数值欧式距离 -> 相似度
        r_num = {k: float(r[k]) if np.isfinite(r[k]) else float("nan") for k in case_num.keys()}

        d1, c1 = weighted_euclidean_distance(case_num, r_num, NUM_W_ACADEMIC, mm_params, missing_policy="penalize")
        d2, c2 = weighted_euclidean_distance(case_num, r_num, NUM_W_ACHIEVEMENT, mm_params, missing_policy="penalize")
        d3, c3 = weighted_euclidean_distance(case_num, r_num, NUM_W_PRACTICE, mm_params, missing_policy="penalize")
        sim1, sim2, sim3 = euclid_to_similarity(d1), euclid_to_similarity(d2), euclid_to_similarity(d3)

        # W4：类别杰卡德（性格/家庭；同时把“专业”作为类别一致性信息单独呈现）
        jac_components: Dict[str, float] = {}
        jac_pf = jaccard_similarity(tokenize_category(case_cat["性格"]), tokenize_category(r["性格"]))
        jac_fam = jaccard_similarity(tokenize_category(case_cat["家庭"]), tokenize_category(r["家庭"]))
        jac_major = jaccard_similarity(tokenize_category(case_cat["专业"]), tokenize_category(r["专业"]))
        jac_components["性格"] = jac_pf
        jac_components["家庭"] = jac_fam
        jac_components["专业"] = jac_major

        # 严格按题设：W4内部(性格/家庭各0.5)
        sim4 = CAT_W_BACKGROUND["性格"] * jac_pf + CAT_W_BACKGROUND["家庭"] * jac_fam

        # 动态视图融合
        sim_total = (
            view_weights["W1"] * sim1
            + view_weights["W2"] * sim2
            + view_weights["W3"] * sim3
            + view_weights["W4"] * sim4
        )

        out.append(
            SimilarityBreakdown(
                id=sid,
                target=target,
                sim_w1=sim1,
                sim_w2=sim2,
                sim_w3=sim3,
                sim_w4=sim4,
                sim_total=sim_total,
                w1_detail=c1,
                w2_detail=c2,
                w3_detail=c3,
                w4_jaccard_components=jac_components,
                missing_key_info=missing_fields,
            )
        )
    return out


# -----------------------------
# 4. 群体筛选 + 群体行为一致性 + 动态路径生成
# -----------------------------

def select_top_k_same_target(
    sims: List[SimilarityBreakdown], goal: str, k: int = 5
) -> List[SimilarityBreakdown]:
    same_goal = [s for s in sims if str(s.target).strip() == str(goal).strip()]
    same_goal.sort(key=lambda x: x.sim_total, reverse=True)
    return same_goal[:k]


def group_behavior_consistency(
    df: pd.DataFrame, top_group_ids: List[str], behavior_fields: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    群体行为一致性（可解释口径）：
      对每个“行为字段”取二值化/离散化后的众数占比，再对字段求平均。
    返回：总体一致性(0~1)，以及每字段一致性明细。
    """
    sub = df[df[ID_COL].astype("string").isin(top_group_ids)].copy()
    details: Dict[str, float] = {}
    if len(sub) == 0:
        return 0.0, {f: 0.0 for f in behavior_fields}

    for f in behavior_fields:
        if f in (NUM_COLS_ACADEMIC + NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE):
            # 行为字段二值化：>0视为“有/参与”，否则“无/未参与”
            s = sub[f]
            b = s.apply(lambda v: 1 if (np.isfinite(v) and float(v) > 0) else 0)
            mode = int(b.mode().iloc[0]) if len(b.mode()) else 0
            acc = float((b == mode).mean())
            details[f] = acc
        else:
            s = sub[f].astype("string").fillna("")
            mode = s.mode().iloc[0] if len(s.mode()) else ""
            acc = float((s == mode).mean())
            details[f] = acc

    overall = float(np.mean(list(details.values()))) if details else 0.0
    return overall, details


def synthesize_dynamic_path(
    df: pd.DataFrame, top_group_ids: List[str], goal: str
) -> List[str]:
    """
    将“相似群体的共性行为”转为动态路径（可用于论文描述）：
    - 以群体中占比>=0.6的关键行动作为“主路径节点”
    - 不同目标使用不同节点优先级（体现“静态结果 -> 动态路径”）
    """
    sub = df[df[ID_COL].astype("string").isin(top_group_ids)].copy()
    if len(sub) == 0:
        return ["未形成可用路径：未检索到同目标相似群体。"]

    # 核心行动字段（可按目标调整）
    if goal == "考研":
        actions = ["最高英语", "GPA", "科研成果", "竞赛", "实习", "志愿时长", "学生职务"]
    elif goal == "保研":
        actions = ["GPA", "科研成果", "竞赛", "学生职务", "最高英语", "志愿时长", "实习"]
    elif goal == "考公":
        actions = ["实习", "学生职务", "志愿时长", "最高英语", "综测", "GPA"]
    elif goal == "留学":
        actions = ["最高英语", "GPA", "科研成果", "竞赛", "实习", "学生职务"]
    else:  # 就业等
        actions = ["实习", "竞赛", "学生职务", "志愿时长", "GPA", "最高英语"]

    path: List[str] = []
    n = len(sub)
    for a in actions:
        if a in NUM_COLS_ACADEMIC:
            # 连续变量：用“均值水平”描述（相对全体分位的论文口径在此简化为均值）
            mean_v = float(pd.to_numeric(sub[a], errors="coerce").mean())
            if np.isfinite(mean_v):
                path.append(f"{a}维持在群体均值水平（均值≈{mean_v:.2f}）")
        elif a in (NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE):
            cnt = int(((sub[a].fillna(0) > 0)).sum())
            ratio = cnt / n if n else 0.0
            if ratio >= 0.6:
                path.append(f"强化{a}（群体中{cnt}/{n}具有该经历，比例={ratio:.0%}）")
            else:
                path.append(f"{a}作为可选增强项（群体中比例={ratio:.0%}，存在分化）")
        else:
            pass

    # 结尾：给出阶段化表达
    if goal in {"考研", "保研"}:
        path.insert(0, "阶段1（基础学业）→ 夯实GPA/综测，并完成英语达标")
        path.append("阶段2（学术与竞争力）→ 以科研/竞赛打造可证明能力")
        path.append("阶段3（实践与复试/导师匹配）→ 适度实践，形成可叙述经历链")
    elif goal == "考公":
        path.insert(0, "阶段1（稳定履历）→ 通过职务/志愿/实习形成公共服务与组织经历")
        path.append("阶段2（能力背书）→ 用综测/学业保持基本盘，英语作为辅助项")
    else:
        path.insert(0, "阶段1（能力与履历）→ 以实习/项目为主线，学业为底线")

    return path


# -----------------------------
# 5. 三支决策：正域/边界域/负域（严格阈值）
# -----------------------------

@dataclass
class ThreeWayDecision:
    domain: str  # 正域/边界域/负域
    reason: str


def three_way_decision(
    top_group: List[SimilarityBreakdown],
    similarity_threshold_negative: float = 0.3,
    pos_consistency_threshold: float = 0.80,
    bnd_consistency_low: float = 0.50,
    bnd_consistency_high: float = 0.80,
    group_consistency: float = 0.0,
    any_missing_key_info: bool = False,
) -> ThreeWayDecision:
    if len(top_group) == 0:
        return ThreeWayDecision(domain="负域", reason="未检索到同目标相似群体，无法形成匹配路径。")

    # 负域：相似度 < 0.3（按题设：相似度＜0.3）
    best_sim = float(top_group[0].sim_total)
    if best_sim < similarity_threshold_negative:
        return ThreeWayDecision(domain="负域", reason=f"最相似样本综合相似度={best_sim:.3f}＜0.3，匹配不足。")

    # 正域：一致性≥80% 且无关键信息缺失
    if (group_consistency >= pos_consistency_threshold) and (not any_missing_key_info):
        return ThreeWayDecision(
            domain="正域",
            reason=f"群体行为一致性={group_consistency:.0%}≥80%，且关键字段无缺失。",
        )

    # 边界域：一致性50%-80%或关键信息缺失
    if (bnd_consistency_low <= group_consistency < bnd_consistency_high) or any_missing_key_info:
        why = []
        if bnd_consistency_low <= group_consistency < bnd_consistency_high:
            why.append(f"群体行为一致性={group_consistency:.0%}处于[50%,80%)")
        if any_missing_key_info:
            why.append("存在关键字段缺失")
        return ThreeWayDecision(domain="边界域", reason="；".join(why) + "，输出路径并提示不确定性。")

    # 其余情况：默认边界域（更保守）
    return ThreeWayDecision(domain="边界域", reason="一致性不足以进入正域，采用保守输出并提示不确定性。")


# -----------------------------
# 6. 论文式报告输出
# -----------------------------

def format_top_group_table(top_group: List[SimilarityBreakdown]) -> str:
    headers = ["Rank", "编号", "最终出路", "Sim(W1)", "Sim(W2)", "Sim(W3)", "Sim(W4)", "Sim(Total)"]
    rows = [headers]
    for i, s in enumerate(top_group, start=1):
        rows.append(
            [
                str(i),
                s.id,
                s.target,
                f"{s.sim_w1:.4f}",
                f"{s.sim_w2:.4f}",
                f"{s.sim_w3:.4f}",
                f"{s.sim_w4:.4f}",
                f"{s.sim_total:.4f}",
            ]
        )
    # markdown table
    md = []
    md.append("| " + " | ".join(rows[0]) + " |")
    md.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def print_report(
    df: pd.DataFrame,
    case: StudentCase,
    sims: List[SimilarityBreakdown],
    top_group: List[SimilarityBreakdown],
    group_cons: float,
    cons_detail: Dict[str, float],
    path: List[str],
    decision: ThreeWayDecision,
):
    vw = DYNAMIC_VIEW_WEIGHTS[case.目标]

    # 选Top1作为“展示详细计算过程”的代表
    exemplar = top_group[0] if top_group else None

    print("\n" + "=" * 88)


def _view_score_breakdown(
    exemplar: SimilarityBreakdown, view_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    将Top-1相似样本的各视图相似度，按动态视图权重折算为“对总分的贡献”。
    用于解释：建议主要针对哪一个视图的短板。
    """
    return {
        "W1": float(view_weights["W1"]) * float(exemplar.sim_w1),
        "W2": float(view_weights["W2"]) * float(exemplar.sim_w2),
        "W3": float(view_weights["W3"]) * float(exemplar.sim_w3),
        "W4": float(view_weights["W4"]) * float(exemplar.sim_w4),
    }


def _missing_penalty_fields(exemplar: SimilarityBreakdown) -> List[str]:
    """
    在当前实现中，缺失惩罚会导致某维的贡献项恰好等于其权重（例如最高英语=0.1、竞赛=0.6等）。
    这里用于识别：Top-1里哪些字段触发了缺失惩罚，从而支撑“边界域/不确定性”的依据解释。
    """
    penalty_fields: List[str] = []
    # W1
    for k, w in NUM_W_ACADEMIC.items():
        v = exemplar.w1_detail.get(k, float("nan"))
        if np.isfinite(v) and abs(float(v) - float(w)) < 1e-9:
            penalty_fields.append(k)
    # W2
    for k, w in NUM_W_ACHIEVEMENT.items():
        v = exemplar.w2_detail.get(k, float("nan"))
        if np.isfinite(v) and abs(float(v) - float(w)) < 1e-9:
            penalty_fields.append(k)
    # W3
    for k, w in NUM_W_PRACTICE.items():
        v = exemplar.w3_detail.get(k, float("nan"))
        if np.isfinite(v) and abs(float(v) - float(w)) < 1e-9:
            penalty_fields.append(k)
    return penalty_fields


def _group_ratio(df: pd.DataFrame, group_ids: List[str], field: str) -> float:
    sub = df[df[ID_COL].astype("string").isin(group_ids)].copy()
    if len(sub) == 0:
        return 0.0
    if field in (NUM_COLS_ACADEMIC + NUM_COLS_ACHIEVEMENT + NUM_COLS_PRACTICE):
        b = (sub[field].fillna(0) > 0).astype(int)
        return float(b.mean())
    s = sub[field].astype("string").fillna("")
    mode = s.mode().iloc[0] if len(s.mode()) else ""
    return float((s == mode).mean())


def build_action_plan(
    df: pd.DataFrame,
    case: StudentCase,
    top_group: List[SimilarityBreakdown],
    decision: ThreeWayDecision,
    group_ids: List[str],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    数据驱动智能体层（不依赖API）：
    - 基于 Top-1 的“视图贡献”定位短板视图
    - 基于 Top 群体真实出现比例（0~1）给出补强动作
    - 输出每条动作的“证据句”（可写进论文的可解释性段落）
    """
    plan: List[Dict[str, str]] = []
    evidence: List[str] = []

    vw = DYNAMIC_VIEW_WEIGHTS[case.目标]
    exemplar = top_group[0] if top_group else None
    if exemplar is not None:
        contrib = _view_score_breakdown(exemplar, vw)
        weakest_view = min(contrib.items(), key=lambda kv: kv[1])[0]
        evidence.append(
            f"Top-1样本(编号={exemplar.id})视图贡献：W1={contrib['W1']:.4f}, W2={contrib['W2']:.4f}, "
            f"W3={contrib['W3']:.4f}, W4={contrib['W4']:.4f}；最低贡献视图={weakest_view}。"
        )
        penalty_fields = _missing_penalty_fields(exemplar)
        if penalty_fields:
            evidence.append(f"Top-1中触发缺失惩罚的字段：{', '.join(penalty_fields)}（缺失会降低相似度并触发不确定性）。")

    # 1) 缺失/不确定性优先：只在确有证据（缺失惩罚字段存在）时才给“补全”建议，避免看起来像模板
    if exemplar is not None:
        penalty_fields = _missing_penalty_fields(exemplar)
        if penalty_fields:
            plan.append(
                {
                    "阶段": "本周",
                    "动作": f"补全并核验关键字段：{', '.join(penalty_fields)}（区分“0”与“未知缺失”）",
                    "指标": "上述字段均有明确取值",
                    "验收": "重新匹配后不再出现缺失惩罚项；三支决策不再仅因缺失进入边界域",
                }
            )

    # 2) 数据驱动补强：用群体占比决定“主线”还是“可选增强”
    # 目标仍然影响优先级，但动作内容由群体统计触发
    action_fields = ["最高英语", "GPA", "科研成果", "竞赛", "实习", "志愿时长", "学生职务"]
    for f in action_fields:
        ratio = _group_ratio(df, group_ids, f) if group_ids else 0.0
        if f in NUM_COLS_ACADEMIC:
            # 连续变量：用Top群体均值给阈值建议
            sub = df[df[ID_COL].astype("string").isin(group_ids)].copy()
            mean_v = float(pd.to_numeric(sub[f], errors="coerce").mean()) if len(sub) else float("nan")
            if np.isfinite(mean_v):
                evidence.append(f"群体统计：{f}均值≈{mean_v:.2f}（用于设定达标阈值/参照线）。")
                plan.append(
                    {
                        "阶段": "1-4周",
                        "动作": f"{f}对齐群体参照线：当前={getattr(case, f) if hasattr(case, f) else 'NA'}，建议至少达到≈{mean_v:.2f}",
                        "指标": f"{f}≥{mean_v:.2f}（或接近群体均值）",
                        "验收": f"{f}达到阈值后重新匹配，观察W1贡献是否提升",
                    }
                )
        else:
            # 二值经历：按占比决定主线/可选
            if ratio >= 0.6:
                plan.append(
                    {
                        "阶段": "1-8周",
                        "动作": f"将{f}作为主线补强（相似群体中占比={ratio:.0%}）",
                        "指标": f"{f}从0/缺失→形成可验证记录/成果",
                        "验收": f"重新匹配后W3/W2相关相似度提升，群体一致性不下降",
                    }
                )
            else:
                plan.append(
                    {
                        "阶段": "可选增强",
                        "动作": f"{f}为可选增强项（相似群体中占比={ratio:.0%}，存在分化）",
                        "指标": "如选择投入，则至少形成1个可验证结果",
                        "验收": "作为个体差异化亮点写入材料并可追溯验证",
                    }
                )
            evidence.append(f"群体统计：{f}出现比例={ratio:.0%}（决定其在路径中的主次）。")

    if decision.domain == "边界域":
        plan.append(
            {
                "阶段": "下次迭代",
                "动作": "按证据定位的短板视图优先补强后进行二次匹配（以Top群体为参照）",
                "指标": "Top-1综合相似度提升，且群体一致性≥80%（若可达）",
                "验收": "三支决策由边界域→正域（或不确定性显著降低）",
            }
        )
    if decision.domain == "负域":
        plan.append(
            {
                "阶段": "1-12周",
                "动作": "夯实基础后重跑匹配（优先学业/英语/实践其一突破）",
                "指标": "关键指标提升（至少一项）",
                "验收": "Top-1综合相似度≥0.3，进入可推荐域",
            }
        )
    return plan, evidence


def render_markdown_report(
    case: StudentCase,
    top_group: List[SimilarityBreakdown],
    group_cons: float,
    cons_detail: Dict[str, float],
    path: List[str],
    decision: ThreeWayDecision,
    action_plan: List[Dict[str, str]],
    evidence: Optional[List[str]] = None,
) -> str:
    vw = DYNAMIC_VIEW_WEIGHTS[case.目标]
    lines: List[str] = []
    lines.append("## 学生发展导航智能体：推荐报告\n\n")
    lines.append("### 1 在读生画像（当前状态）\n")
    for k, v in case.to_dict().items():
        lines.append(f"- **{k}**：{v}\n")
    lines.append("\n### 2 目标驱动动态视图权重\n")
    lines.append(f"- **W1/W2/W3/W4** = {vw['W1']}/{vw['W2']}/{vw['W3']}/{vw['W4']}\n")
    lines.append("\n### 3 相似群体（同目标Top-k）\n")
    if top_group:
        lines.append(format_top_group_table(top_group) + "\n")
    else:
        lines.append("- 未检索到同目标相似群体。\n")
    lines.append("\n### 4 群体一致性与动态路径\n")
    lines.append(f"- **群体一致性**：{group_cons:.0%}\n")
    for k, v in cons_detail.items():
        lines.append(f"  - {k}: {v:.0%}\n")
    lines.append("\n**动态路径**：\n")
    for i, s in enumerate(path, start=1):
        lines.append(f"{i}. {s}\n")
    lines.append("\n### 5 三支决策\n")
    lines.append(f"- **决策域**：{decision.domain}\n- **依据**：{decision.reason}\n")
    lines.append("\n### 6 行动计划（可执行闭环）\n")
    for item in action_plan:
        lines.append(
            f"- **阶段**：{item['阶段']}\n  - **动作**：{item['动作']}\n  - **指标**：{item['指标']}\n  - **验收**：{item['验收']}\n"
        )
    if evidence:
        lines.append("\n### 7 建议依据（可解释性证据）\n")
        for e in evidence:
            lines.append(f"- {e}\n")
    return "".join(lines)


class ArkLLM:
    """
    可选：ARK/火山 LLM 话术层（不强依赖）。
    - 不在代码里保存任何Key；只从环境变量读取
    - 若调用失败，外层应自动降级为规则/模板输出
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout_s: int = 60,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_s = timeout_s

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:  # noqa: BLE001
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"ARK调用失败 HTTP {getattr(e, 'code', 'NA')}: {detail}") from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"ARK调用失败: {type(e).__name__}: {e}") from e

        obj = json.loads(raw)
        # 兼容常见OpenAI风格响应
        try:
            return obj["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"ARK响应解析失败：{raw[:500]}") from e


def nlg_with_llm_or_fallback(
    *,
    case: StudentCase,
    top_group: List[SimilarityBreakdown],
    group_cons: float,
    cons_detail: Dict[str, float],
    path: List[str],
    decision: ThreeWayDecision,
    action_plan: List[Dict[str, str]],
    evidence: List[str],
    enable_llm: bool,
    ark_base_url: str,
) -> Tuple[str, Optional[str]]:
    """
    返回：(markdown_report, llm_text_or_none)
    - markdown_report：结构化Markdown（永远可用，便于论文粘贴）
    - llm_text_or_none：若启用LLM且调用成功，返回LLM生成的“论文式话术段落”；否则None
    """
    md = render_markdown_report(
        case=case,
        top_group=top_group,
        group_cons=group_cons,
        cons_detail=cons_detail,
        path=path,
        decision=decision,
        action_plan=action_plan,
        evidence=evidence,
    )

    if not enable_llm:
        return md, None

    api_key = os.getenv("ARK_API_KEY", "").strip()
    model_name = os.getenv("VOLC_MODEL_NAME", "").strip() or "deepseek-v3"
    if not api_key:
        return md, None

    llm = ArkLLM(api_key=api_key, model_name=model_name, base_url=ark_base_url)
    system_prompt = (
        "你是学术写作助手。请把给定的结构化实验结果，改写为可直接放入论文“实验与结果分析”章节的中文段落。"
        "要求：使用学术化表述，逻辑清晰；必须引用证据（Top群体编号/相似度/一致性/边界域原因）；不要编造数据。"
    )
    user_prompt = (
        "下面是结构化结果（Markdown）。请生成：\n"
        "1) 1段“实验设置与方法要点”总结；\n"
        "2) 1段“结果与分析”（含Top群体、相似度、三支决策域、动态路径要点、不确定性来源）；\n"
        "3) 1段“建议与讨论”（围绕证据与行动计划，强调数据驱动与可复现）。\n\n"
        f"{md}"
    )
    text = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    return md, text


class NavigationAgent:
    """
    交互式智能体外壳（命令行多轮）：
    - 主动追问并区分“0”与“未知缺失”
    - 维护状态（case），支持save/load
    - 每轮重算：群体匹配 + 三支决策 + 动态路径 + 行动计划
    """

    def __init__(self, df: pd.DataFrame, case: StudentCase):
        self.df = df
        self.case = case
        self.last_report_md: str = ""

    def recompute(self) -> Tuple[List[SimilarityBreakdown], float, Dict[str, float], List[str], ThreeWayDecision, List[Dict[str, str]]]:
        sims = compute_similarity(
            df=self.df,
            case=self.case,
            view_weights=DYNAMIC_VIEW_WEIGHTS[self.case.目标],
            key_fields_for_missing=[
                "高考",
                "综测",
                "GPA",
                "最高英语",
                "竞赛",
                "科研成果",
                "实习",
                "志愿时长",
                "学生职务",
                "性格",
                "家庭",
                "专业",
            ],
        )
        top_group = select_top_k_same_target(sims, goal=self.case.目标, k=5)
        behavior_fields = ["竞赛", "科研成果", "实习", "志愿时长", "学生职务", "性格", "家庭", "专业"]
        group_ids = [s.id for s in top_group]
        cons, cons_detail = group_behavior_consistency(self.df, group_ids, behavior_fields)
        any_missing = any(len(s.missing_key_info) > 0 for s in top_group)
        path = synthesize_dynamic_path(self.df, group_ids, goal=self.case.目标)
        decision = three_way_decision(top_group=top_group, group_consistency=cons, any_missing_key_info=any_missing)
        action_plan, evidence = build_action_plan(
            df=self.df, case=self.case, top_group=top_group, decision=decision, group_ids=group_ids
        )
        # 交互模式下：默认不强制调用LLM（避免卡住/收费）；用户可用 /export 导出后再走report+llm
        self.last_report_md, _ = nlg_with_llm_or_fallback(
            case=self.case,
            top_group=top_group,
            group_cons=cons,
            cons_detail=cons_detail,
            path=path,
            decision=decision,
            action_plan=action_plan,
            evidence=evidence,
            enable_llm=False,
            ark_base_url=os.getenv("ARK_BASE_URL", "").strip()
            or "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
        return top_group, cons, cons_detail, path, decision, action_plan

    @staticmethod
    def _ask(prompt: str) -> str:
        return input(prompt).strip()

    @staticmethod
    def _ask_optional_float(prompt: str) -> Optional[float]:
        while True:
            s = NavigationAgent._ask(prompt)
            if s == "":
                return None
            try:
                return float(s)
            except ValueError:
                print("请输入数字；未知可直接回车跳过。")

    @staticmethod
    def _ask_optional_str(prompt: str) -> Optional[str]:
        s = NavigationAgent._ask(prompt)
        return None if s == "" else s

    def interact(self):
        print("\n=== 学生发展导航智能体（Agent交互模式）===")
        print("指令：/show 查看状态；/save <json>；/load <json>；/export <md>；/quit 退出。\n")

        _, _, _, _, decision, _ = self.recompute()
        print(f"当前决策域：{decision.domain}（目标={self.case.目标}）")

        while True:
            cmd = self._ask("\n> ")
            if cmd == "/quit":
                print("已退出。")
                return
            if cmd.startswith("/show"):
                print(json.dumps(self.case.to_dict(), ensure_ascii=False, indent=2))
                continue
            if cmd.startswith("/save"):
                parts = cmd.split(maxsplit=1)
                path = parts[1] if len(parts) == 2 else "agent_state.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.case.to_dict(), f, ensure_ascii=False, indent=2)
                print(f"已保存：{path}")
                continue
            if cmd.startswith("/load"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print("用法：/load <file.json>")
                    continue
                path = parts[1]
                with open(path, "r", encoding="utf-8") as f:
                    self.case = StudentCase.from_dict(json.load(f))
                print(f"已载入：{path}")
                _, _, _, _, decision, _ = self.recompute()
                print(f"更新后决策域：{decision.domain}")
                continue
            if cmd.startswith("/export"):
                parts = cmd.split(maxsplit=1)
                path = parts[1] if len(parts) == 2 else "agent_report.md"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.last_report_md)
                print(f"已导出：{path}")
                continue

            # 默认：进入一轮“补全信息->重算->反馈”
            print("进入补全环节（回车=跳过；填0表示确实为0而非缺失）。")

            goal = self._ask_optional_str(f"目标（当前={self.case.目标}；考研/保研/考公/留学/就业）= ")
            if goal:
                goal = goal.strip()
                if goal in DYNAMIC_VIEW_WEIGHTS:
                    self.case = StudentCase.from_dict({**self.case.to_dict(), "目标": goal})
                else:
                    print("目标不在可选范围内，已忽略。")

            # 关键字段优先问（与你当前输出的“缺失触发边界域”一致）
            updates: Dict[str, Any] = {}
            for f in ["最高英语", "竞赛", "科研成果", "实习", "学生职务"]:
                cur = self.case.to_dict()[f]
                v = self._ask_optional_float(f"{f}（当前={cur}）= ")
                if v is not None:
                    updates[f] = float(v)
            for f in ["性格", "家庭", "专业"]:
                cur = self.case.to_dict()[f]
                v = self._ask_optional_str(f"{f}（当前={cur}）= ")
                if v is not None:
                    updates[f] = str(v)
            if updates:
                self.case = StudentCase.from_dict({**self.case.to_dict(), **updates})

            top_group, cons, _, path, decision, action_plan = self.recompute()
            print("\n--- 本轮反馈（摘要）---")
            print(f"决策域：{decision.domain}")
            print(f"依据：{decision.reason}")
            if top_group:
                print(f"Top相似群体编号：{', '.join([s.id for s in top_group])}")
                print(f"Top-1综合相似度：{top_group[0].sim_total:.4f}")
            print(f"群体一致性：{cons:.0%}")
            print("动态路径（前3步）：")
            for s in path[:3]:
                print(f"- {s}")
            print("行动计划（前2项）：")
            for item in action_plan[:2]:
                print(f"- {item['阶段']}：{item['动作']}（验收：{item['验收']}）")


# -----------------------------
# 7. 主程序：按给定案例运行（题设案例可直接复现）
# -----------------------------

def main():
    # 先尝试从本地 .env 载入密钥/配置（如果存在）
    _load_dotenv_if_exists()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data.csv", help="data.csv路径（默认同目录）")
    ap.add_argument("--target", default="考研", help="目标：考研/保研/考公/留学/就业")
    ap.add_argument("--encoding", default=None, help="可选：强制指定CSV编码（如 gb18030 / gbk / utf-8-sig 等）")
    ap.add_argument("--mode", default="report", help="运行模式：report(一次性论文式报告) / agent(交互式智能体)")
    ap.add_argument("--export_md", default=None, help="可选：导出Markdown报告路径（report模式）")
    ap.add_argument("--nlg", default="rule", help="话术生成：rule(默认，规则+证据) / llm(调用ARK生成论文段落)")
    ap.add_argument(
        "--ark_base_url",
        default=os.getenv("ARK_BASE_URL", "").strip() or "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        help="ARK ChatCompletions URL（可用环境变量ARK_BASE_URL覆盖）",
    )
    args = ap.parse_args()

    df = load_dataset(args.csv, encoding=args.encoding)

    # 题设在读生案例（可直接用于论文“实验与结果分析”）
    case = StudentCase(
        专业="数科",
        高考=578,
        综测=77,
        GPA=3.25,
        最高英语=468,  # 四级468（数值化）
        竞赛=0,
        科研成果=0,
        实习=0,
        志愿时长=38,
        学生职务=0,
        性格="偏内向",
        家庭="一般",
        目标=str(args.target).strip(),
    )
    if case.目标 not in DYNAMIC_VIEW_WEIGHTS:
        raise ValueError(f"不支持的目标：{case.目标}，可选={list(DYNAMIC_VIEW_WEIGHTS.keys())}")

    if str(args.mode).strip().lower() == "agent":
        NavigationAgent(df=df, case=case).interact()
        return

    sims = compute_similarity(
        df=df,
        case=case,
        view_weights=DYNAMIC_VIEW_WEIGHTS[case.目标],
        key_fields_for_missing=[
            "高考",
            "综测",
            "GPA",
            "最高英语",
            "竞赛",
            "科研成果",
            "实习",
            "志愿时长",
            "学生职务",
            "性格",
            "家庭",
            "专业",
        ],
    )

    top_group = select_top_k_same_target(sims, goal=case.目标, k=5)

    # 一致性字段建议（与“路径节点”强相关）
    behavior_fields = ["竞赛", "科研成果", "实习", "志愿时长", "学生职务", "性格", "家庭", "专业"]
    group_ids = [s.id for s in top_group]
    cons, cons_detail = group_behavior_consistency(df, group_ids, behavior_fields)

    any_missing = any(len(s.missing_key_info) > 0 for s in top_group)
    path = synthesize_dynamic_path(df, group_ids, goal=case.目标)
    decision = three_way_decision(
        top_group=top_group,
        group_consistency=cons,
        any_missing_key_info=any_missing,
    )

    print_report(df, case, sims, top_group, cons, cons_detail, path, decision)

    if args.export_md:
        action_plan, evidence = build_action_plan(
            df=df, case=case, top_group=top_group, decision=decision, group_ids=group_ids
        )
        md, _ = nlg_with_llm_or_fallback(
            case=case,
            top_group=top_group,
            group_cons=cons,
            cons_detail=cons_detail,
            path=path,
            decision=decision,
            action_plan=action_plan,
            evidence=evidence,
            enable_llm=str(args.nlg).strip().lower() == "llm",
            ark_base_url=str(args.ark_base_url).strip(),
        )
        with open(args.export_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n[已导出Markdown报告] {args.export_md}")


if __name__ == "__main__":
    main()

