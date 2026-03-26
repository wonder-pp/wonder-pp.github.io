import re
from typing import Dict, List


class UserProfiler:
    def parse_user_input(self, user_input: str) -> Dict:
        text = user_input.strip()
        profile = {
            "原始输入": text,
            "当前阶段": self._extract_stage(text),
            "目标": self._extract_goal(text),
            "GPA原始值": self._extract_gpa_raw(text),
            "GPA等级": self._extract_gpa_level(text),
            "科研强度": self._extract_research_intensity(text),
            "竞赛强度": self._extract_competition_intensity(text),
            "语言水平": self._extract_language_level(text),
            "关注问题": self._extract_concerns(text),
            "偏好方向": self._extract_preferences(text),
        }
        profile["缺失信息"] = self._find_missing_fields(profile)
        profile["检索查询"] = self.build_retrieval_query(profile)
        profile["画像摘要"] = self.build_profile_summary(profile)
        return profile

    def build_retrieval_query(self, profile: Dict) -> str:
        parts: List[str] = [profile["当前阶段"], profile["目标"]]

        if profile["科研强度"] == 0:
            parts.append("科研起步")
        elif profile["科研强度"] >= 3:
            parts.append("科研提升")

        if profile["竞赛强度"] == 0:
            parts.append("竞赛起步")
        elif profile["竞赛强度"] == 1:
            parts.append("竞赛准备")
        elif profile["竞赛强度"] >= 3:
            parts.append("竞赛进阶")

        parts.extend(profile["关注问题"][:2])
        parts.append("经验建议")

        deduped_parts: List[str] = []
        for part in parts:
            if part and part != "未知" and part not in deduped_parts:
                deduped_parts.append(part)
        return " ".join(deduped_parts)

    def build_profile_summary(self, profile: Dict) -> str:
        summary = (
            f"用户目前处于{profile['当前阶段']}，目标是{profile['目标']}，"
            f"GPA等级为{profile['GPA等级']}，科研强度为{profile['科研强度']}，"
            f"竞赛强度为{profile['竞赛强度']}，语言水平为{profile['语言水平']}。"
        )
        if profile["关注问题"]:
            summary += f" 当前更关心：{'、'.join(profile['关注问题'])}。"
        if profile["偏好方向"]:
            summary += f" 偏好方向：{'、'.join(profile['偏好方向'])}。"
        return summary

    def _extract_stage(self, text: str) -> str:
        for stage in ["大一", "大二", "大三", "大四"]:
            if stage in text:
                return stage

        grade_match = re.search(r"20(\d{2})\s*级", text)
        if grade_match:
            year = int(f"20{grade_match.group(1)}")
            stage_map = {
                2025: "大一",
                2024: "大二",
                2023: "大三",
                2022: "大四",
            }
            return stage_map.get(year, "未知")
        return "未知"

    def _extract_goal(self, text: str) -> str:
        text_lower = text.lower()
        goal_map = {
            "保研": ["保研", "推免", "直博", "夏令营", "预推免"],
            "考研": ["考研", "研究生", "硕士", "复试"],
            "留学": ["留学", "出国", "海外", "雅思", "托福", "gre", "gmat"],
            "就业": ["就业", "找工作", "求职", "实习", "秋招", "春招", "offer"],
            "考公": ["考公", "公务员", "选调", "行测", "申论"],
        }
        for goal, keywords in goal_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return goal
        return "未知"

    def _extract_gpa_raw(self, text: str) -> str:
        for pattern in [r"gpa[^0-9]*(\d+(?:\.\d+)?)", r"绩点[^0-9]*(\d+(?:\.\d+)?)"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        for keyword in ["优秀", "良好", "一般", "偏低"]:
            if keyword in text:
                return keyword
        return ""

    def _extract_gpa_level(self, text: str) -> int:
        raw_value = self._extract_gpa_raw(text)
        number_match = re.search(r"\d+(?:\.\d+)?", raw_value)
        if number_match:
            gpa = float(number_match.group())
            if gpa >= 3.7:
                return 3
            if gpa >= 3.0:
                return 2
            return 1

        if any(keyword in text for keyword in ["优秀", "很高", "前10%", "前 10%", "前10名"]):
            return 3
        if any(keyword in text for keyword in ["良好", "中上", "前30%", "前 30%"]):
            return 2
        if any(keyword in text for keyword in ["一般", "普通", "偏低"]):
            return 1
        return 2

    def _extract_research_intensity(self, text: str) -> int:
        if any(keyword in text for keyword in ["没有科研", "无科研", "零科研"]):
            return 0
        if any(keyword in text for keyword in ["一点科研", "少量科研", "刚开始做科研"]):
            return 1
        if any(keyword in text for keyword in ["论文", "SCI", "EI", "专利", "发表", "一作"]):
            return 3
        if any(keyword in text for keyword in ["科研项目", "大创", "实验室", "课题", "科研"]):
            return 2
        return 0

    def _extract_competition_intensity(self, text: str) -> int:
        if any(keyword in text for keyword in ["没有竞赛", "无竞赛", "零竞赛"]):
            return 0
        if any(keyword in text for keyword in ["国家级", "国奖", "一等奖", "全国"]):
            return 3
        if any(keyword in text for keyword in ["省级", "市级", "二等奖", "校级一等奖"]):
            return 2
        if any(keyword in text for keyword in ["竞赛", "比赛", "建模"]):
            return 1
        return 0

    def _extract_language_level(self, text: str) -> str:
        if any(keyword in text for keyword in ["雅思7", "雅思 7", "托福100", "托福 100", "六级", "CET6"]):
            return "高"
        if any(keyword in text for keyword in ["四级", "CET4", "雅思", "托福"]):
            return "中"
        if any(keyword in text for keyword in ["英语一般", "英语弱", "英语不好"]):
            return "低"
        return "未知"

    def _extract_concerns(self, text: str) -> List[str]:
        concern_keywords = {
            "时间规划": ["规划", "时间", "安排", "路线"],
            "科研准备": ["科研", "论文", "实验室", "项目"],
            "竞赛准备": ["竞赛", "比赛", "建模"],
            "材料准备": ["简历", "材料", "文书", "面试", "夏令营", "复试"],
            "绩点提升": ["GPA", "gpa", "绩点", "成绩", "排名"],
            "实习就业": ["实习", "求职", "秋招", "春招", "offer"],
        }
        concerns = [name for name, keywords in concern_keywords.items() if any(keyword in text for keyword in keywords)]
        return concerns

    def _extract_preferences(self, text: str) -> List[str]:
        preferences = []
        if any(keyword in text for keyword in ["想去高校", "读研", "学术", "科研型"]):
            preferences.append("学术导向")
        if any(keyword in text for keyword in ["就业导向", "尽快工作", "找工作"]):
            preferences.append("就业导向")
        if any(keyword in text for keyword in ["稳一点", "保守", "求稳"]):
            preferences.append("稳妥路径")
        if any(keyword in text for keyword in ["冲一冲", "冲刺", "挑战"]):
            preferences.append("冲刺路径")
        return preferences

    def _find_missing_fields(self, profile: Dict) -> List[str]:
        missing = []
        if profile["当前阶段"] == "未知":
            missing.append("当前阶段")
        if profile["目标"] == "未知":
            missing.append("目标")
        if not profile["GPA原始值"]:
            missing.append("GPA")
        if profile["语言水平"] == "未知":
            missing.append("语言水平")
        return missing


if __name__ == "__main__":
    profiler = UserProfiler()
    sample = "我大二，GPA 3.4，没有科研，有一点竞赛，想保研，主要想知道怎么做规划和准备材料"
    print(profiler.parse_user_input(sample))
