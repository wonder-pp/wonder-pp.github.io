"""
数据层：高校数智学院往届学生数据预处理

目标：
- 支持 Excel / CSV 读取
- 将定性特征量化为数值（性格、发展意向等）
- 进行缺失值与简单异常值处理
- 输出适配 `ImprovedCFMatcher` 的特征矩阵
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# 1. 字段量化规则（可按学院实际调整）
# -------------------------

PERSONALITY_MAP = {
    "外向": 4,
    "偏外向": 3,
    "中性": 2,
    "偏内向": 1.5,
    "内向": 1,
}

FAMILY_BACKGROUND_MAP = {
    "困难": 1,
    "一般": 2,
    "良好": 3,
    "优渥": 4,
}

DEVELOPMENT_INTENTION_MAP = {
    "保研": 1,
    "考研": 2,
    "留学": 3,
    "考公": 4,
    "就业": 5,
}


def _read_any(path: str) -> pd.DataFrame:
    """
    读取 Excel / CSV：
    - Excel：直接用 pandas.read_excel
    - CSV：模仿 0312.py 中的鲁棒读法，自动尝试多种常见中文编码，
      避免 UnicodeDecodeError（例如 GBK / GB2312 数据集）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # 尝试多种编码
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    
    # 如果CSV读取失败，尝试作为Excel读取（有时文件扩展名可能不对）
    try:
        return pd.read_excel(path)
    except Exception as e:  # noqa: BLE001
        pass
    
    raise RuntimeError(
        f"无法以常见编码读取CSV：{path}；最后错误={type(last_err).__name__}: {last_err}"
    )


def quantize_and_clean(
    path: str,
    id_col: str = "学生编号",
    outcome_col: str = "最终出路",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取往届学生 Excel/CSV，并做量化与清洗。

    返回：
    - df_q: 含新建的数值化特征列
    - feature_cols: 推荐算法使用的特征列名列表
    """
    df = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]

    if id_col not in df.columns:
        df[id_col] = range(1, len(df) + 1)

    # ---- 1) 性格量化 ----
    def map_personality(x: object) -> float:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        return float(PERSONALITY_MAP.get(s, 2))

    df["性格倾向评分"] = df.get("性格", "").apply(map_personality)

    # ---- 2) 家庭背景量化 ----
    def map_family(x: object) -> float:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        return float(FAMILY_BACKGROUND_MAP.get(s, 2))

    df["家庭背景评分"] = df.get("家庭背景", df.get("家庭", "")).apply(map_family)

    # ---- 3) 成绩水平量化：示例用GPA/综测/高考统一到[0,4]相对评分 ----
    # 若数据集中没有可用字段，可自由调整权重
    def _safe_to_num(series_name: str) -> pd.Series:
        if series_name not in df.columns:
            return pd.Series([np.nan] * len(df))
        return pd.to_numeric(df[series_name], errors="coerce")

    gpa = _safe_to_num("GPA")
    zongce = _safe_to_num("综测")
    gaokao = _safe_to_num("高考")

    # 归一到0~1后再映射到0~4
    def norm01(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            return pd.Series([np.nan] * len(s))
        return (s - mn) / (mx - mn)

    score_gpa = norm01(gpa)
    score_zc = norm01(zongce)
    score_gk = norm01(gaokao)
    df["成绩水平评分"] = (
        0.5 * score_gpa.fillna(score_gpa.mean())
        + 0.3 * score_zc.fillna(score_zc.mean())
        + 0.2 * score_gk.fillna(score_gk.mean())
    ) * 4.0

    # ---- 4) 发展意向量化 ----
    def map_dev_intent(x: object) -> float:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        return float(DEVELOPMENT_INTENTION_MAP.get(s, 0))

    if "发展意向" in df.columns:
        df["发展意向编码"] = df["发展意向"].apply(map_dev_intent)
    else:
        # 若无专门“发展意向”，可用“最终出路”近似
        df["发展意向编码"] = df[outcome_col].apply(map_dev_intent)

    # ---- 5) 竞赛经历量化（示例：次数+层次综合）----
    # 若已有“竞赛分数/等级”，直接数值化；否则以是否有经历作为0/1
    if "竞赛" in df.columns:
        comp = pd.to_numeric(df["竞赛"], errors="coerce")
        comp_norm = norm01(comp).fillna(0.0) * 4.0
        df["竞赛经历评分"] = comp_norm
    else:
        df["竞赛经历评分"] = 0.0

    # ---- 6) 简单异常值处理：对数值列做3σ截断 ----
    num_cols = [
        "家庭背景评分",
        "性格倾向评分",
        "成绩水平评分",
        "发展意向编码",
        "竞赛经历评分",
    ]
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mu, sigma = s.mean(skipna=True), s.std(skipna=True)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0:
            continue
        upper, lower = mu + 3 * sigma, mu - 3 * sigma
        s = s.clip(lower=lower, upper=upper)
        df[c] = s

    feature_cols = num_cols
    return df, feature_cols


def build_new_student_feature(
    *,
    family: str,
    personality: str,
    score_level: float,
    dev_intent: str,
    competition_level: Optional[float],
) -> Dict[str, float]:
    """
    帮助函数：将前端表单输入转换为特征字典。
    """
    family_score = FAMILY_BACKGROUND_MAP.get(family.strip(), 2)
    pers_score = PERSONALITY_MAP.get(personality.strip(), 2)
    dev_score = DEVELOPMENT_INTENTION_MAP.get(dev_intent.strip(), 0)

    return {
        "家庭背景评分": float(family_score),
        "性格倾向评分": float(pers_score),
        "成绩水平评分": float(score_level),
        "发展意向编码": float(dev_score),
        "竞赛经历评分": (
            float(competition_level) if competition_level is not None else np.nan
        ),
    }


__all__ = [
    "quantize_and_clean",
    "build_new_student_feature",
    "PERSONALITY_MAP",
    "FAMILY_BACKGROUND_MAP",
    "DEVELOPMENT_INTENTION_MAP",
]

