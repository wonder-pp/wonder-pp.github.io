import os
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

from prompt_builder import PromptBuilder


class AnswerGenerator:
    def __init__(self):
        load_dotenv()
        self.prompt_builder = PromptBuilder()
        self.api_key = (
            os.getenv("ARK_API_KEY")
            or os.getenv("VOLC_API_KEY")
            or os.getenv("DOUBAO_API_KEY")
            or ""
        )
        self.base_url = os.getenv(
            "ARK_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
        self.model = (
            os.getenv("ARK_MODEL")
            or os.getenv("VOLC_MODEL_NAME")
            or os.getenv("MODEL_NAME")
            or ""
        )

    def generate_answer(
        self,
        user_profile: Dict,
        top_matches: List[Dict],
        goal_matches: List[Dict],
        retrieved_experiences: List[Dict],
        path_template: Dict[str, List[str]],
    ) -> Tuple[str, str]:
        prompt_preview = self.prompt_builder.build_preview(
            user_profile,
            top_matches,
            goal_matches,
            retrieved_experiences,
            path_template,
        )

        if self.api_key and self.model:
            try:
                messages = self.prompt_builder.build_messages(
                    user_profile,
                    top_matches,
                    goal_matches,
                    retrieved_experiences,
                    path_template,
                )
                return self._call_llm(messages), prompt_preview
            except Exception as exc:
                print(f"LLM 调用失败，回退到规则生成: {exc}")

        return (
            self._build_fallback_answer(
                user_profile,
                top_matches,
                goal_matches,
                retrieved_experiences,
                path_template,
            ),
            prompt_preview,
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.4,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _build_fallback_answer(
        self,
        user_profile: Dict,
        top_matches: List[Dict],
        goal_matches: List[Dict],
        retrieved_experiences: List[Dict],
        path_template: Dict[str, List[str]],
    ) -> str:
        lines = []
        lines.append("## 路径判断")
        lines.append(user_profile["画像摘要"])
        if user_profile["缺失信息"]:
            lines.append(f"当前还缺少 {'、'.join(user_profile['缺失信息'])}，因此部分判断需要保守解读。")

        lines.append("\n## 相似案例")
        if top_matches:
            for item in top_matches[:3]:
                lines.append(
                    f"- {item.get('显示名称', '匿名案例')}：目标{item['目标']}，"
                    f"GPA等级{item['GPA等级']}，科研强度{item['科研强度']}，"
                    f"竞赛强度{item['竞赛强度']}，相似度 {item['相似度']:.3f}"
                )
        else:
            lines.append("- 暂无足够相似的案例。")

        lines.append("\n## 差距分析")
        if goal_matches:
            ref = goal_matches[0]
            has_gap = False
            if user_profile["GPA等级"] < ref["GPA等级"]:
                lines.append("- 你的成绩竞争力仍有提升空间。")
                has_gap = True
            if user_profile["科研强度"] < ref["科研强度"]:
                lines.append("- 你的科研积累弱于同目标参考案例。")
                has_gap = True
            if user_profile["竞赛强度"] < ref["竞赛强度"]:
                lines.append("- 你的竞赛成果还可以继续补强。")
                has_gap = True
            if not has_gap:
                lines.append("- 你和同目标参考案例的核心画像差距不大，更关键的是持续输出。")
        else:
            lines.append("- 暂无同目标案例，建议先参考相似案例。")

        lines.append("\n## 行动建议")
        stage_actions = path_template.get(user_profile["当前阶段"], [])
        if stage_actions:
            for action in stage_actions[:5]:
                lines.append(f"- {action}")
        else:
            lines.append("- 当前阶段模板不足，建议先补充更完整背景信息。")

        lines.append("\n## 经验引用")
        if retrieved_experiences:
            for item in retrieved_experiences[:2]:
                lines.append(f"- {item.get('显示名称', '匿名案例')}：{item['chunk_text']}")
        else:
            lines.append("- 暂未检索到直接相关的经验原文。")

        return "\n".join(lines)
