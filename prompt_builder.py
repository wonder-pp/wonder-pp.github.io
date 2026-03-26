from typing import Dict, List


class PromptBuilder:
    def build_messages(
        self,
        user_profile: Dict,
        top_matches: List[Dict],
        goal_matches: List[Dict],
        retrieved_experiences: List[Dict],
        path_template: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_prompt(
                    user_profile,
                    top_matches,
                    goal_matches,
                    retrieved_experiences,
                    path_template,
                ),
            },
        ]

    def build_preview(
        self,
        user_profile: Dict,
        top_matches: List[Dict],
        goal_matches: List[Dict],
        retrieved_experiences: List[Dict],
        path_template: Dict[str, List[str]],
    ) -> str:
        messages = self.build_messages(
            user_profile,
            top_matches,
            goal_matches,
            retrieved_experiences,
            path_template,
        )
        return "\n\n".join([f"[{item['role']}]\n{item['content']}" for item in messages])

    def _build_system_prompt(self) -> str:
        return (
            "你是一个学业规划顾问。请基于匿名化的真实采访经验片段生成建议。\n"
            "要求：\n"
            "1. 不要暴露真实姓名，不要自行猜测身份。\n"
            "2. 经验引用必须只基于给定片段，不要编造额外经历。\n"
            "3. 优先从经验中提炼可执行动作，而不是复述采访提问。\n"
            "4. 如果经验片段不足以支撑某个判断，要明确说明不确定。\n"
            "5. 输出结构为：路径判断、相似案例、差距分析、行动建议、经验引用。"
        )

    def _build_user_prompt(
        self,
        user_profile: Dict,
        top_matches: List[Dict],
        goal_matches: List[Dict],
        retrieved_experiences: List[Dict],
        path_template: Dict[str, List[str]],
    ) -> str:
        sections = [
            "一、用户画像",
            self._format_profile(user_profile),
            "二、相似案例",
            self._format_matches(top_matches),
            "三、同目标案例",
            self._format_matches(goal_matches),
            "四、匿名经验片段",
            self._format_experiences(retrieved_experiences),
            "五、阶段模板建议",
            self._format_path_template(user_profile, path_template),
            "六、生成要求",
            (
                "请基于以上信息生成最终建议：\n"
                "- 先判断这条路径是否适合当前用户。\n"
                "- 再总结相似案例与同目标案例的共性。\n"
                "- 接着指出当前最关键短板。\n"
                "- 最后给出 3 到 5 条具体可执行建议。\n"
                "- 至少引用 1 条经验片段，并解释为什么与当前用户相关。\n"
                "- 不要出现真实姓名，不要把采访问题原样写进结果。"
            ),
        ]
        return "\n\n".join(sections)

    def _format_profile(self, user_profile: Dict) -> str:
        return "\n".join(
            [
                f"- 原始输入：{user_profile['原始输入']}",
                f"- 当前阶段：{user_profile['当前阶段']}",
                f"- 目标：{user_profile['目标']}",
                f"- GPA原始值：{user_profile['GPA原始值'] or '未明确'}",
                f"- GPA等级：{user_profile['GPA等级']}",
                f"- 科研强度：{user_profile['科研强度']}",
                f"- 竞赛强度：{user_profile['竞赛强度']}",
                f"- 语言水平：{user_profile['语言水平']}",
                f"- 关注问题：{'、'.join(user_profile['关注问题']) if user_profile['关注问题'] else '未明确'}",
                f"- 偏好方向：{'、'.join(user_profile['偏好方向']) if user_profile['偏好方向'] else '未明确'}",
                f"- 缺失信息：{'、'.join(user_profile['缺失信息']) if user_profile['缺失信息'] else '无'}",
                f"- 画像摘要：{user_profile['画像摘要']}",
            ]
        )

    def _format_matches(self, matches: List[Dict]) -> str:
        if not matches:
            return "- 暂无案例"
        lines = []
        for item in matches[:3]:
            lines.append(
                f"- {item.get('显示名称', '匿名案例')} | 年级: {item.get('标准年级', item.get('年级', '未知'))} | "
                f"目标: {item.get('目标', '未知')} | GPA等级: {item.get('GPA等级', '未知')} | "
                f"科研强度: {item.get('科研强度', '未知')} | 竞赛强度: {item.get('竞赛强度', '未知')} | "
                f"相似度: {item.get('相似度', 0):.3f}"
            )
        return "\n".join(lines)

    def _format_experiences(self, experiences: List[Dict]) -> str:
        if not experiences:
            return "- 暂无经验片段"
        lines = []
        for item in experiences[:3]:
            lines.append(
                f"- {item.get('显示名称', '匿名案例')} | 目标: {item.get('目标', '未知')} | "
                f"分数: {item.get('score', 0):.3f} | 片段: {item.get('chunk_text', '')}"
            )
        return "\n".join(lines)

    def _format_path_template(self, user_profile: Dict, path_template: Dict[str, List[str]]) -> str:
        actions = path_template.get(user_profile["当前阶段"], [])
        if not actions:
            return "- 暂无阶段模板"
        return "\n".join([f"- {action}" for action in actions])
