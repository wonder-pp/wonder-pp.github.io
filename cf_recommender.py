"""
高校数智学院学生学业导航：改进协同过滤核心算法

对应论文：岳希等(2020)《基于数据稀疏性的协同过滤推荐算法改进研究》

本文件聚焦“算法层Baseline”，实现三点改进：
1) 缺失值填补：平均值 + 相似用户预测值（对应论文的“邻域加权预测填补”）
2) 加权相似度：在传统余弦相似度基础上，增加“共同特征数”权重修正（缓解数据稀疏）
3) 用户-项目双维度相似度融合：同时考虑“学生间相似度”和“特征/项目间相似度”

注意：这里把“项目/Item”抽象为学生发展相关的特征维度
（如：竞赛强度、科研水平、实践经历、最终出路等），
以便在高校学业导航场景中落地。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StudentProfile:
    """
    单个学生的结构化特征向量封装。

    必须保证 `features` 与特征矩阵中的列同顺序。
    """

    student_id: str
    features: np.ndarray  # 包含数值化后的全部特征（可能包含NaN）
    meta: Dict[str, str]  # 例如：姓名、专业、最终结果、发展路径描述等


@dataclass
class MatchResult:
    """
    推荐结果：Top-K相似学生。
    """

    student_id: str
    similarity: float
    common_feature_ratio: float
    user_sim: float
    item_sim: float
    meta: Dict[str, str]


class ImprovedCFMatcher:
    """
    面向“数智学院学业导航”的改进协同过滤匹配器。

    对应岳希等(2020)三点改进：
    - 改进1：缺失值填补（平均值 + 相似用户预测） -> `_impute_missing_iterative`
    - 改进2：考虑共同特征权重的相似度 -> `_user_similarity_with_common_weight`
    - 改进3：用户-项目双维融合 -> `match_top_k` 中 `user_sim` 与 `item_sim` 融合
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = "学生编号",
        outcome_col: str = "最终出路",
        max_impute_iter: int = 3,
        common_weight_alpha: float = 0.5,
        user_item_lambda: float = 0.7,
    ) -> None:
        """
        参数说明：
        - historical_df: 往届学生原始数据（已完成基础清洗与量化；含缺失/稀疏）
        - feature_cols: 参与相似度计算的特征列（数值化后）
        - id_col: 学生唯一标识列
        - outcome_col: 发展结果/意向列，可作为“项目维度”的一部分
        - max_impute_iter: 缺失值迭代填补轮数（过大收益递减，默认3）
        - common_weight_alpha: 共同特征权重影响系数 \\(0~1\\)
        - user_item_lambda: 用户-项目双维相似度融合系数
          sim_final = λ * sim_user + (1-λ) * sim_item
        """
        self.id_col = id_col
        self.outcome_col = outcome_col
        self.feature_cols = feature_cols
        self.max_impute_iter = max_impute_iter
        self.common_weight_alpha = common_weight_alpha
        self.user_item_lambda = user_item_lambda

        self.raw_df = historical_df.copy()
        self._build_matrices()

    # ------------------------------------------------------------------
    # 1. 构建特征矩阵 & 基础统计
    # ------------------------------------------------------------------
    def _build_matrices(self) -> None:
        df = self.raw_df
        self.student_ids = df[self.id_col].astype("string").tolist()
        self.outcomes = df[self.outcome_col].astype("string").tolist()

        feat = df[self.feature_cols].astype("float64")
        self.feature_matrix = feat.to_numpy(dtype=float)  # shape: (n_students, n_features)

        # 均值用于改进1的第一步填补
        self.feature_means = np.nanmean(self.feature_matrix, axis=0)
        self.feature_means = np.where(
            np.isfinite(self.feature_means), self.feature_means, 0.0
        )

    # ------------------------------------------------------------------
    # 2. 改进1：平均值 + 相似用户预测 填补缺失
    # ------------------------------------------------------------------
    def _impute_missing_iterative(self) -> None:
        """
        对往届学生特征矩阵进行缺失值填补。

        岳希等改进思路：
        - 首先用全局或局部均值粗略填补，得到初始完整矩阵；
        - 然后基于该矩阵计算用户相似度，利用邻域加权预测进一步修正缺失位置。
        """
        X = self.feature_matrix.copy()
        n_users, n_items = X.shape

        # Step 1: 用全局均值粗填补
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(self.feature_means, np.where(nan_mask)[1])

        for _ in range(self.max_impute_iter):
            # 计算当前用户相似度（不加共同特征权重，只作为填补依据）
            sim_u = self._cosine_similarity(X)

            # 对每个缺失位置，用“相似用户的加权平均”更新
            for u in range(n_users):
                for i in range(n_items):
                    if not nan_mask[u, i]:
                        continue
                    # 选取在该特征上非缺失的相似用户
                    neighbors_mask = ~nan_mask[:, i]
                    neighbors_mask[u] = False
                    if not np.any(neighbors_mask):
                        # 若该特征几乎全缺失，则退回全局均值
                        X[u, i] = self.feature_means[i]
                        continue
                    sims = sim_u[u, neighbors_mask]
                    values = X[neighbors_mask, i]
                    # 只使用相似度>0的邻居
                    positive = sims > 0
                    if not np.any(positive):
                        X[u, i] = self.feature_means[i]
                        continue
                    sims = sims[positive]
                    values = values[positive]
                    X[u, i] = float(np.dot(sims, values) / np.sum(sims))

        self.imputed_feature_matrix = X

    # ------------------------------------------------------------------
    # 3. 改进2：加入“共同特征数”权重的用户相似度
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(mat: np.ndarray) -> np.ndarray:
        # 标准余弦相似度：不显式处理稀疏，但已在填补阶段缓解
        norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_normed = mat / norm
        return mat_normed @ mat_normed.T

    def _user_similarity_with_common_weight(
        self, X: np.ndarray, mask_valid: np.ndarray
    ) -> np.ndarray:
        """
        在余弦相似度基础上，乘以“共同特征比例”的权重项。

        对应岳希等“通过共同评分项目数修正用户相似度”的思路，
        在学生学业导航场景下，等价为“共同非缺失特征比例”。
        """
        base_sim = self._cosine_similarity(X)
        n_users, n_features = mask_valid.shape

        # common_count[u,v] = 两人都非缺失的特征数
        common = mask_valid.astype(int) @ mask_valid.astype(int).T
        max_common = float(n_features)
        common_ratio = common / max_common  # 0~1

        # 共同特征权重：w = (common_ratio)^alpha
        weight = np.power(common_ratio, self.common_weight_alpha)
        return base_sim * weight

    # ------------------------------------------------------------------
    # 4. 改进3：用户-项目双维相似度
    # ------------------------------------------------------------------
    def _item_similarity(self, X: np.ndarray) -> np.ndarray:
        """
        计算“特征/项目间”的相似度，用于用户-项目双维融合。

        在学业导航中，可以理解为：
        - 哪些发展特征（GPA/竞赛/科研/实践/出路等）在群体中呈现相似模式。
        """
        # 对特征矩阵转置后做余弦相似度：Shape -> (n_features, n_features)
        return self._cosine_similarity(X.T)

    # ------------------------------------------------------------------
    # 5. 对新学生进行Top-K匹配
    # ------------------------------------------------------------------
    def match_top_k(
        self,
        new_student_features: Dict[str, float],
        top_k: int = 5,
    ) -> List[MatchResult]:
        """
        输入：
        - new_student_features: 新学生的特征字典（可能含缺失/None）
        - top_k: 返回最相似的往届学生数量

        输出：
        - MatchResult 列表（按最终综合相似度从高到低排序）
        """
        # 1) 确保历史特征已完成缺失填补
        if not hasattr(self, "imputed_feature_matrix"):
            self._impute_missing_iterative()

        X_hist = self.imputed_feature_matrix  # (n_users, n_features)
        n_users, n_features = X_hist.shape

        # 2) 构造新学生特征向量（允许部分缺失）
        x_new = np.empty((n_features,), dtype=float)
        valid_mask_new = np.zeros((n_features,), dtype=bool)
        for j, col in enumerate(self.feature_cols):
            v = new_student_features.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                # 用历史均值粗填补；但保留“缺失”信息给共同特征权重使用
                x_new[j] = self.feature_means[j]
                valid_mask_new[j] = False
            else:
                x_new[j] = float(v)
                valid_mask_new[j] = True

        # 3) 计算用户相似度（改进2）
        X_all = np.vstack([X_hist, x_new[None, :]])  # 把新学生也拼进去
        valid_mask_all = np.vstack(
            [~np.isnan(self.feature_matrix), valid_mask_new[None, :]]
        )
        sim_user_all = self._user_similarity_with_common_weight(X_all, valid_mask_all)
        sim_user_new = sim_user_all[-1, :-1]  # 新学生与每个历史学生的相似度

        # 4) 计算项目/特征相似度（Item-based CF）
        S_item = self._item_similarity(X_all)
        # 利用特征相似度，构造“新学生在各特征上的隐含偏好向量”
        # 这里采用简单线性组合：p_new = x_new @ S_item
        p_new = x_new @ S_item  # Shape: (n_features,)
        # 再将每个历史学生的特征向量与 p_new 做余弦相似度，得到 item-based user 相似度
        norm_hist = np.linalg.norm(X_hist, axis=1) + 1e-8
        norm_p = np.linalg.norm(p_new) + 1e-8
        sim_item = (X_hist @ p_new) / (norm_hist * norm_p)

        # 5) 用户-项目双维融合（改进3）
        sim_final = (
            self.user_item_lambda * sim_user_new
            + (1.0 - self.user_item_lambda) * sim_item
        )

        # 6) 计算共同特征比例（便于解释“数据稀疏性”的影响）
        valid_hist = ~np.isnan(self.feature_matrix)
        common = valid_hist & valid_mask_new[None, :]
        common_ratio = common.sum(axis=1) / float(n_features)

        # 7) 排序并包装输出
        order = np.argsort(-sim_final)  # 降序
        results: List[MatchResult] = []
        for idx in order[:top_k]:
            meta = {
                "最终出路": str(self.outcomes[idx]),
            }
            # 将原始行中的其它字段也一并放入meta（如：发展路径文本）
            row = self.raw_df.iloc[idx]
            for c in row.index:
                if c in (self.id_col, self.outcome_col):
                    continue
                if c in self.feature_cols:
                    continue
                meta[str(c)] = str(row[c])

            results.append(
                MatchResult(
                    student_id=str(self.student_ids[idx]),
                    similarity=float(sim_final[idx]),
                    common_feature_ratio=float(common_ratio[idx]),
                    user_sim=float(sim_user_new[idx]),
                    item_sim=float(sim_item[idx]),
                    meta=meta,
                )
            )
        return results


def demo_match():
    """
    简单命令行测试入口：
    - 读取 demo_historical.csv
    - 构造一个新学生特征
    - 打印Top-3相似学生
    """
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "demo_historical.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"示例文件 {csv_path} 不存在，请先准备一份往届学生数据。"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    feature_cols = [
        "家庭背景评分",
        "性格倾向评分",
        "成绩水平评分",
        "发展意向编码",
        "竞赛经历评分",
    ]

    matcher = ImprovedCFMatcher(
        historical_df=df,
        feature_cols=feature_cols,
        id_col="学生编号",
        outcome_col="最终出路",
    )

    new_student = {
        "家庭背景评分": 2.0,
        "性格倾向评分": 3.0,
        "成绩水平评分": 3.5,
        "发展意向编码": 2.0,  # 例如：考研=2
        "竞赛经历评分": np.nan,  # 未知/缺失
    }

    results = matcher.match_top_k(new_student, top_k=3)
    print("\n=== Top-3 相似学生（示例）===")
    for r in results:
        print(
            f"学生 {r.student_id} | 综合相似度={r.similarity:.4f} "
            f"(UserSim={r.user_sim:.4f}, ItemSim={r.item_sim:.4f}, "
            f"共同特征比例={r.common_feature_ratio:.2%}) | 最终出路={r.meta.get('最终出路')}"
        )


if __name__ == "__main__":
    demo_match()

