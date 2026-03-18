"""
极简版 Streamlit 前端：
- 用户填写基本画像（家庭、性格、成绩、发展意向、竞赛）
- 点击"匹配"按钮，调用改进协同过滤算法得到 Top-K 相似学生
- 展示匹配结果 + LLM 个性化建议
- 绘制特征对比雷达图
- 提供"发展规划清单"下载
"""

from __future__ import annotations

import io
import os
from typing import List, Dict

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cf_recommender import ImprovedCFMatcher
from data_processing import (
    PERSONALITY_MAP,
    FAMILY_BACKGROUND_MAP,
    DEVELOPMENT_INTENTION_MAP,
    build_new_student_feature,
    quantize_and_clean,
)
from llm_client import build_default_llm


def _load_historical_dataset() -> tuple[pd.DataFrame, List[str]]:
    """
    读取/预处理往届学生数据。
    默认使用 `demo_historical.csv`，也允许用户在侧边栏上传。
    """
    st.sidebar.subheader("数据源配置")
    uploaded = st.sidebar.file_uploader("上传往届学生数据 (CSV/Excel)", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        tmp_path = os.path.join(os.getcwd(), "_tmp_uploaded.csv")
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        df_q, feature_cols = quantize_and_clean(tmp_path)
        return df_q, feature_cols

    # 默认示例数据
    default_path = os.path.join(os.path.dirname(__file__), "demo_historical.csv")
    if not os.path.exists(default_path):
        st.error("未找到默认示例数据 demo_historical.csv，请在项目根目录放置该文件。")
        st.stop()

    df_q, feature_cols = quantize_and_clean(default_path)
    return df_q, feature_cols


def radar_chart_compare(
    new_vec: np.ndarray,
    hist_vec: np.ndarray,
    feature_cols: List[str],
    title: str,
):
    """
    绘制新学生与某个相似学生的特征雷达图。
    """
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    labels = feature_cols
    num_vars = len(labels)

    # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    new_vals = new_vec.tolist()
    hist_vals = hist_vec.tolist()
    new_vals += new_vals[:1]
    hist_vals += hist_vals[:1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles, new_vals, "o-", linewidth=2, label="新学生")
    ax.fill(angles, new_vals, alpha=0.25)
    ax.plot(angles, hist_vals, "o-", linewidth=2, label="相似学生")
    ax.fill(angles, hist_vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)


def build_planning_checklist_with_llm(
    matches,
    dev_intent: str,
    new_features: Dict[str, float],
    llm_client,
) -> str:
    """
    使用LLM生成完全个性化的"发展规划清单"Markdown文本。
    基于学生画像、匹配结果和发展意向生成针对性的四阶段建议。
    """
    if not matches:
        return "# 学业发展规划清单\n\n未找到足够的相似学生，可先从夯实基础学业做起。\n"

    # 构建学生画像描述
    student_profile = {
        "家庭背景评分": new_features.get("家庭背景评分", "未知"),
        "性格倾向评分": new_features.get("性格倾向评分", "未知"),
        "成绩水平评分": new_features.get("成绩水平评分", "未知"),
        "发展意向": dev_intent,
        "竞赛经历评分": new_features.get("竞赛经历评分", "未知"),
    }

    # 构建相似案例信息
    match_info = []
    for i, m in enumerate(matches[:3]):  # 取前3个最相似的案例
        match_info.append({
            "排名": i + 1,
            "学生编号": m.student_id,
            "最终出路": m.meta.get("最终出路", "未知"),
            "综合相似度": round(m.similarity, 3),
            "用户相似度": round(m.user_sim, 3),
            "项目相似度": round(m.item_sim, 3),
        })

    # 构建LLM提示
    system_prompt = (
        "你是一名资深的高校学业生涯规划导师，擅长根据学生画像和相似案例生成个性化的发展规划。"
        "请根据提供的学生特征和相似成功案例，生成一份详细、具体、可执行的四阶段发展规划。"
        "每个阶段的建议必须结合学生的具体情况，不能是通用模板。"
    )

    user_prompt = f"""
    【学生画像】
    {student_profile}
    
    【发展意向】
    {dev_intent}
    
    【参考案例（Top-3相似学生）】
    {match_info}
    
    【要求】
    请生成一份Markdown格式的学业发展规划清单，包含以下内容：
    
    1. **目标方向分析**：基于学生画像和相似案例，分析该学生选择{dev_intent}的优劣势
    
    2. **四阶段具体行动建议**：
       - 阶段1（基础夯实期）：根据当前成绩水平，给出具体的学业提升目标和方法
       - 阶段2（能力提升期）：根据性格特点和发展意向，给出针对性的能力培养建议
       - 阶段3（实践积累期）：根据竞赛经历和相似案例，给出具体的竞赛/科研/实践路径
       - 阶段4（冲刺准备期）：根据发展意向，给出具体的申请/考试准备策略和时间表
    
    3. **关键成功因素**：基于相似案例，总结该学生需要特别关注的关键因素
    
    4. **风险提示**：基于学生画像，提醒可能遇到的风险和应对策略
    
    请确保建议具体、可执行，并且紧密结合学生的实际情况，避免使用通用模板语言。
    """

    try:
        # 调用LLM生成个性化建议
        response = llm_client.generate_plan(
            new_student_profile=student_profile,
            matches=match_info,
            user_question=user_prompt,
        )
        
        # 如果LLM调用成功，返回生成的内容
        if response and not response.startswith("【"):
            return f"# 学业发展规划清单\n\n{response}"
    except Exception as e:
        pass

    # 如果LLM调用失败，返回基础信息
    best = matches[0]
    lines: List[str] = []
    lines.append("# 学业发展规划清单\n\n")
    lines.append(f"- 目标方向：{dev_intent}\n")
    lines.append(
        f"- 参考案例：学生 {best.student_id}（最终出路：{best.meta.get('最终出路', '未知')}，"
        f"综合相似度≈{best.similarity:.3f}）\n\n"
    )
    lines.append("## 建议分阶段行动\n\n")
    lines.append("### 阶段1：基础学业与成绩\n")
    lines.append("- 明确绩点/综合测评的目标区间，并按学期拆解到具体科目。\n")
    lines.append("- 针对薄弱课程安排查缺补漏时间（每周至少2次专题复盘）。\n\n")
    lines.append("### 阶段2：英语与核心能力\n")
    lines.append("- 设定英语分数/等级达标时间（如大二结束前通过四级/六级）。\n")
    lines.append("- 结合目标方向，在写作、表达或科研阅读上做专项训练。\n\n")
    lines.append("### 阶段3：竞赛 / 科研 / 实践\n")
    lines.append("- 选择1–2条主线（如专业竞赛、导师科研、行业实习），避免过度分散。\n")
    lines.append("- 为每条主线设定可验证成果（奖项、论文、项目、实习证明等）。\n\n")
    lines.append("### 阶段4：申请与冲刺\n")
    lines.append("- 梳理材料（个人陈述、简历、项目总结），用数据和案例讲清成长路径。\n")
    lines.append("- 根据目标（保研/考研/留学/考公）安排报名、笔试/面试时间表。\n")
    return "".join(lines)


def main():
    st.set_page_config(page_title="高校数智学院学业导航智能体", layout="wide")
    st.title("高校数智学院学业导航智能体（协同过滤 + LLM）")

    st.markdown(
        "本Demo实现了：**改进协同过滤匹配 + 缺失值填补 + LLM个性化建议 + 雷达图对比 + 规划清单下载**。"
    )

    df_hist, feature_cols = _load_historical_dataset()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("一、填写基本画像")
        family = st.selectbox("家庭背景", list(FAMILY_BACKGROUND_MAP.keys()))
        personality = st.selectbox("性格倾向", list(PERSONALITY_MAP.keys()))
        score_level = st.slider("整体成绩水平估计（0=较弱, 4=较强）", 0.0, 4.0, 2.5, 0.1)
        dev_intent = st.selectbox("发展意向", list(DEVELOPMENT_INTENTION_MAP.keys()))
        has_comp = st.checkbox("有一定竞赛/科创经历", value=False)
        comp_level = (
            st.slider("竞赛/科创强度（0=无, 4=很强）", 0.0, 4.0, 2.0, 0.1)
            if has_comp
            else None
        )
        user_question = st.text_area(
            "自述 / 追问（可选）",
            "例如：我想考研但目前竞赛科研几乎为0，该如何规划？",
        )

        top_k = st.slider("Top-K 相似学生数量", 3, 10, 5)

        run_btn = st.button("开始匹配与规划")

    if not run_btn:
        return

    # 在主区域显示所有结果
    st.markdown("---")
    st.markdown("## 💡 匹配结果与建议")

    # 构造新学生特征
    new_features = build_new_student_feature(
        family=family,
        personality=personality,
        score_level=score_level,
        dev_intent=dev_intent,
        competition_level=comp_level,
    )

    matcher = ImprovedCFMatcher(
        historical_df=df_hist,
        feature_cols=feature_cols,
        id_col="学生编号",
        outcome_col="最终出路",
    )

    matches = matcher.match_top_k(new_features, top_k=top_k)

    st.markdown("### 1. Top-K 相似学生概览")
    
    # 调试信息
    st.write(f"DEBUG: 找到 {len(matches) if matches else 0} 个匹配结果")
    
    if not matches:
        st.warning("未找到足够的历史样本，请检查数据集。")
        st.write(f"DEBUG: new_features = {new_features}")
        st.write(f"DEBUG: df_hist shape = {df_hist.shape}")
        return

    # 构建Markdown表格以避免PyArrow依赖
    table_md = "| 学生编号 | 综合相似度 | 用户相似度 | 项目相似度 | 共同特征比例 | 最终出路 |\n"
    table_md += "|---|---|---|---|---|---|\n"
    for r in matches:
        table_md += f"| {r.student_id} | {round(r.similarity, 4)} | {round(r.user_sim, 4)} | {round(r.item_sim, 4)} | {r.common_feature_ratio:.0%} | {r.meta.get('最终出路', '')} |\n"
    st.markdown(table_md)

    st.markdown("### 2. 特征对比雷达图（新学生 vs Top-1）")
    best = matches[0]
    idx = df_hist[df_hist["学生编号"].astype(str) == best.student_id].index[0]
    new_vec = np.array([new_features[c] for c in feature_cols], dtype=float)
    hist_vec = df_hist.loc[idx, feature_cols].astype(float).to_numpy()

    radar_chart_compare(
            new_vec=new_vec,
            hist_vec=hist_vec,
            feature_cols=feature_cols,
            title=f"新学生 vs 学生 {best.student_id}",
        )

    st.markdown("### 3. LLM 个性化发展建议")
    with st.spinner("正在调用 Doubao 生成个性化建议..."):
        llm = build_default_llm("doubao")
        new_profile = {
            "家庭背景": family,
            "性格": personality,
            "成绩水平(0-4)": score_level,
            "发展意向": dev_intent,
        }
        match_payload = [
            {
                "学生编号": m.student_id,
                "综合相似度": m.similarity,
                "用户相似度": m.user_sim,
                "项目相似度": m.item_sim,
                "共同特征比例": m.common_feature_ratio,
                "最终出路": m.meta.get("最终出路", ""),
                "其他信息": m.meta,
            }
            for m in matches
        ]
        text = llm.generate_plan(
            new_student_profile=new_profile,
            matches=match_payload,
            user_question=user_question,
        )
    
    if text and len(text) > 0 and not text.startswith("【"):
        st.success("✅ 建议已生成")
        st.markdown(text)
    else:
        st.warning(f"⚠️ LLM 返回信息: {text}")

    st.markdown("### 4. 发展规划清单")
    with st.spinner("正在生成个性化发展规划清单..."):
        checklist_md = build_planning_checklist_with_llm(
            matches=matches,
            dev_intent=dev_intent,
            new_features=new_features,
            llm_client=llm,
        )
    
    # 先显示清单内容
    st.markdown(checklist_md)
    
    # 再提供下载按钮
    st.markdown("---")
    buf = io.BytesIO(checklist_md.encode("utf-8"))
    st.download_button(
        label="📥 下载规划清单 (Markdown)",
        data=buf,
        file_name="学业发展规划清单.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()

