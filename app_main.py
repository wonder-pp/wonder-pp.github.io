import streamlit as st

from answer_generation import AnswerGenerator
from experience_retrieval import ExperienceRetriever
from path_templates import PathTemplate
from peer_matching import PeerMatcher
from privacy_utils import clean_experience_text
from user_profiling import UserProfiler


st.set_page_config(page_title="学业规划智能助手", page_icon="🧭", layout="wide")


def get_services():
    return (
        UserProfiler(),
        PeerMatcher("processed_students.csv"),
        ExperienceRetriever("processed_students.csv"),
        PathTemplate(),
        AnswerGenerator(),
    )


profiler, matcher, retriever, template, generator = get_services()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(216, 232, 255, 0.9), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 232, 214, 0.85), transparent 24%),
                linear-gradient(180deg, #f6f1e8 0%, #eef3f7 52%, #f8fafb 100%);
        }
        .hero {
            padding: 28px 32px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(12, 46, 68, 0.96), rgba(36, 92, 84, 0.93));
            color: #f8f5ef;
            box-shadow: 0 18px 40px rgba(16, 36, 52, 0.16);
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 18px;
        }
        .hero h1 {
            margin: 0 0 10px 0;
            font-size: 2.2rem;
            letter-spacing: 0.02em;
        }
        .hero p {
            margin: 0;
            max-width: 780px;
            color: rgba(248, 245, 239, 0.9);
            line-height: 1.65;
            font-size: 1.02rem;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
        }
        .chip {
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.13);
            color: #fff7ed;
            font-size: 0.9rem;
            border: 1px solid rgba(255,255,255,0.12);
        }
        .section-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(27, 55, 71, 0.08);
            border-radius: 20px;
            padding: 20px 22px;
            box-shadow: 0 14px 30px rgba(42, 57, 70, 0.06);
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #173042;
            margin-bottom: 12px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(245,248,250,0.92));
            border: 1px solid rgba(27, 55, 71, 0.08);
            border-radius: 18px;
            padding: 16px;
            min-height: 122px;
            box-shadow: 0 10px 20px rgba(42, 57, 70, 0.05);
        }
        .metric-label {
            color: #5d6f7e;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #173042;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 8px;
            line-height: 1.35;
            word-break: break-word;
        }
        .metric-note {
            color: #516271;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .match-card, .quote-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(27, 55, 71, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .match-topline, .quote-topline {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            margin-bottom: 10px;
            color: #173042;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #e5efe8;
            color: #23513d;
            font-size: 0.82rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .badge.warm {
            background: #f6e7d7;
            color: #8a4d17;
        }
        .soft-text {
            color: #597081;
            line-height: 1.65;
        }
        .summary-panel {
            background: linear-gradient(135deg, rgba(244, 232, 214, 0.92), rgba(255, 255, 255, 0.95));
            border: 1px solid rgba(146, 101, 52, 0.12);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 10px 24px rgba(95, 68, 35, 0.08);
        }
        .sidebar-note {
            padding: 14px;
            border-radius: 16px;
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(27, 55, 71, 0.08);
            color: #48616e;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>EduPilot数智领航·学业全周期规划专家</h1>
            <p>
                把你的个人情况转成结构化画像，再结合真实学生案例、经验片段检索和路径模板，
                生成一份更像咨询结果页的学业规划建议。
            </p>
            <div class="chip-row">
                <span class="chip">画像解析</span>
                <span class="chip">相似案例匹配</span>
                <span class="chip">经验原文检索</span>
                <span class="chip">Prompt 可调试</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_card(item: dict) -> None:
    summary = clean_experience_text(item.get("经验分享", "暂无经验分享"))
    if len(summary) > 180:
        summary = summary[:180] + "..."
    st.markdown(
        f"""
        <div class="match-card">
            <div class="match-topline">
                <div><strong>{item.get('显示名称', '匿名案例')}</strong></div>
                <div><span class="badge">相似度 {item.get('相似度', 0):.3f}</span></div>
            </div>
            <div style="margin-bottom: 10px;">
                <span class="badge warm">{item.get('目标', '未知目标')}</span>
                <span class="badge">{item.get('标准年级', item.get('年级', '未知年级'))}</span>
                <span class="badge">GPA {item.get('GPA等级', '未知')}</span>
                <span class="badge">科研 {item.get('科研强度', '未知')}</span>
                <span class="badge">竞赛 {item.get('竞赛强度', '未知')}</span>
            </div>
            <div class="soft-text">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"查看 {item.get('显示名称', '匿名案例')} 的完整案例"):
        st.markdown(
            f"**基本信息**\n\n"
            f"- 目标：{item.get('目标', '未知')}\n"
            f"- 年级：{item.get('标准年级', item.get('年级', '未知'))}\n"
            f"- GPA 等级：{item.get('GPA等级', '未知')}\n"
            f"- 科研强度：{item.get('科研强度', '未知')}\n"
            f"- 竞赛强度：{item.get('竞赛强度', '未知')}\n"
            f"- 相似度：{item.get('相似度', 0):.3f}"
        )
        research_text = clean_experience_text(item.get("科研经历", ""))
        competition_text = clean_experience_text(item.get("竞赛", item.get("竞赛经历", "")))
        experience_text = clean_experience_text(item.get("经验分享", ""))
        if research_text:
            st.markdown(f"**科研经历**\n\n{research_text}")
        if competition_text:
            st.markdown(f"**竞赛经历**\n\n{competition_text}")
        if experience_text:
            st.markdown(f"**经验分享**\n\n{experience_text}")


def render_quote_card(item: dict) -> None:
    st.markdown(
        f"""
        <div class="quote-card">
            <div class="quote-topline">
                <div><strong>{item.get('显示名称', '匿名案例')}</strong></div>
                <div>
                    <span class="badge warm">{item.get('目标', '未知目标')}</span>
                    <span class="badge">命中 {item.get('score', 0):.3f}</span>
                </div>
            </div>
            <div class="soft-text">{item.get('chunk_text', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_grid(user_profile: dict) -> None:
    items = [
        ("当前阶段", user_profile["当前阶段"]),
        ("目标", user_profile["目标"]),
        ("GPA原始值", user_profile["GPA原始值"] or "未明确"),
        ("GPA等级", str(user_profile["GPA等级"])),
        ("科研强度", str(user_profile["科研强度"])),
        ("竞赛强度", str(user_profile["竞赛强度"])),
        ("语言水平", user_profile["语言水平"]),
        ("检索查询", user_profile["检索查询"]),
    ]
    col1, col2 = st.columns(2)
    for idx, (label, value) in enumerate(items):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(
                f"""
                <div class="match-card" style="margin-bottom:12px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    render_styles()
    render_hero()

    st.sidebar.markdown("### 数据概览")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            当前已载入 <strong>{len(matcher.df)}</strong> 条学生记录，<strong>{len(retriever.chunks)}</strong> 个文本块。<br>
            输入一段自然语言描述，系统会自动给出画像、案例和建议。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not matcher.df.empty:
        st.sidebar.markdown("### 目标分布")
        goal_counts = matcher.df["目标"].value_counts()
        for goal, count in goal_counts.items():
            st.sidebar.write(f"- {goal}: {count}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">输入你的情况</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "",
        placeholder="例如：我大二，GPA 3.4，没有科研，有一点竞赛，想保研，比较担心时间规划和材料准备。",
        height=130,
        label_visibility="collapsed",
    )
    generate = st.button("生成规划结果页", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not generate:
        return
    if not user_input.strip():
        st.warning("请先输入你的情况。")
        return

    with st.spinner("正在整理画像、匹配案例并生成建议..."):
        user_profile = profiler.parse_user_input(user_input)
        top_matches = matcher.match_top_k(user_profile, top_k=5)
        goal_matches = matcher.match_by_goal(user_profile, top_k=5)
        retrieved_experiences = retriever.retrieve(user_profile["检索查询"], top_k=3)
        path_template = template.get_full_path(user_profile["目标"])
        answer, prompt_preview = generator.generate_answer(
            user_profile,
            top_matches,
            goal_matches,
            retrieved_experiences,
            path_template,
        )

    st.markdown("### 结果总览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("当前阶段", user_profile["当前阶段"], "系统根据你的输入自动判断所在阶段。")
    with col2:
        render_metric_card("目标路径", user_profile["目标"], "这会决定优先参考哪类学生案例。")
    with col3:
        render_metric_card("检索查询", user_profile["检索查询"], "用于经验检索的自动生成查询。")
    with col4:
        missing = "、".join(user_profile["缺失信息"]) if user_profile["缺失信息"] else "信息较完整"
        render_metric_card("信息完整度", "待补充项", missing)

    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">用户画像</div>', unsafe_allow_html=True)
        render_profile_grid(user_profile)
        st.markdown(
            f"""
            <div class="summary-panel">
                <div class="section-title" style="margin-bottom: 8px;">画像摘要</div>
                <div class="soft-text">{user_profile["画像摘要"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if user_profile["关注问题"] or user_profile["偏好方向"]:
            tags = user_profile["关注问题"] + user_profile["偏好方向"]
            st.caption("关注重点")
            st.write(" | ".join(tags))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">经验原文引用</div>', unsafe_allow_html=True)
        if retrieved_experiences:
            for item in retrieved_experiences:
                render_quote_card(item)
        else:
            st.info("暂时没有检索到可直接引用的经验片段。")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        tab1, tab2, tab3 = st.tabs(["综合建议", "相似案例", "Prompt 调试"])

        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">规划建议</div>', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">相似学生参考</div>', unsafe_allow_html=True)
            if top_matches:
                for item in top_matches[:3]:
                    render_match_card(item)
            else:
                st.info("暂时没有找到相似学生。")

            st.markdown('<div class="section-title" style="margin-top: 14px;">同目标参考</div>', unsafe_allow_html=True)
            if goal_matches:
                for item in goal_matches[:2]:
                    render_match_card(item)
            else:
                st.info("暂时没有找到同目标学生。")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">模块 2 Prompt 预览</div>', unsafe_allow_html=True)
            st.code(prompt_preview, language="markdown")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
