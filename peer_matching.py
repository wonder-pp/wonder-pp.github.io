from typing import Dict, List

import pandas as pd

from privacy_utils import anonymize_label


class PeerMatcher:
    def __init__(self, processed_data_path: str = "processed_students.csv"):
        self.df = pd.read_csv(processed_data_path).fillna("")
        self.df = self.df[self.df["目标"] != "未知"].copy()
        print(f"已加载 {len(self.df)} 条可匹配学生记录。")

    def calculate_similarity(self, user_profile: Dict, student: Dict) -> float:
        weights = {
            "GPA等级": 0.4,
            "科研强度": 0.3,
            "竞赛强度": 0.2,
            "年级匹配": 0.1,
        }

        gpa_diff = abs(int(user_profile["GPA等级"]) - int(student.get("GPA等级", 0)))
        research_diff = abs(int(user_profile["科研强度"]) - int(student.get("科研强度", 0)))
        competition_diff = abs(int(user_profile["竞赛强度"]) - int(student.get("竞赛强度", 0)))

        student_stage = self._normalize_student_stage(student.get("年级", ""))
        stage_match = 1.0 if user_profile["当前阶段"] == student_stage else 0.5
        stage_diff = 1.0 - stage_match

        distance = (
            weights["GPA等级"] * gpa_diff
            + weights["科研强度"] * research_diff
            + weights["竞赛强度"] * competition_diff
            + weights["年级匹配"] * stage_diff
        )
        return 1.0 / (1.0 + distance)

    def match_top_k(self, user_profile: Dict, top_k: int = 5) -> List[Dict]:
        matches = []
        for _, row in self.df.iterrows():
            student = row.to_dict()
            student["相似度"] = self.calculate_similarity(user_profile, student)
            student["标准年级"] = self._normalize_student_stage(student.get("年级", ""))
            student["显示名称"] = anonymize_label(student)
            matches.append(student)

        matches.sort(key=lambda item: item["相似度"], reverse=True)
        return matches[:top_k]

    def match_by_goal(self, user_profile: Dict, top_k: int = 5) -> List[Dict]:
        filtered = self.df[self.df["目标"] == user_profile["目标"]]
        matches = []
        for _, row in filtered.iterrows():
            student = row.to_dict()
            student["相似度"] = self.calculate_similarity(user_profile, student)
            student["标准年级"] = self._normalize_student_stage(student.get("年级", ""))
            student["显示名称"] = anonymize_label(student)
            matches.append(student)

        matches.sort(key=lambda item: item["相似度"], reverse=True)
        return matches[:top_k]

    def _normalize_student_stage(self, value: str) -> str:
        text = str(value).strip()
        if any(keyword in text for keyword in ["大一", "一年级"]):
            return "大一"
        if any(keyword in text for keyword in ["大二", "二年级"]):
            return "大二"
        if any(keyword in text for keyword in ["大三", "三年级"]):
            return "大三"
        if any(keyword in text for keyword in ["大四", "四年级"]):
            return "大四"

        year_map = {
            "2025": "大一",
            "2024": "大二",
            "2023": "大三",
            "2022": "大四",
        }
        for year_prefix, stage in year_map.items():
            if text.startswith(year_prefix):
                return stage
        return text or "未知"


if __name__ == "__main__":
    matcher = PeerMatcher()
    profile = {
        "GPA等级": 2,
        "科研强度": 1,
        "竞赛强度": 1,
        "当前阶段": "大二",
        "目标": "保研",
    }
    print(matcher.match_top_k(profile, top_k=3))
