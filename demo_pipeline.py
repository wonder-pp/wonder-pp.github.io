import argparse
import json

from answer_generation import AnswerGenerator
from experience_retrieval import ExperienceRetriever
from path_templates import PathTemplate
from peer_matching import PeerMatcher
from user_profiling import UserProfiler


def run_demo(user_input: str) -> dict:
    profiler = UserProfiler()
    matcher = PeerMatcher("processed_students.csv")
    retriever = ExperienceRetriever("processed_students.csv")
    template = PathTemplate()
    generator = AnswerGenerator()

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

    return {
        "user_profile": user_profile,
        "top_matches": top_matches[:3],
        "goal_matches": goal_matches[:3],
        "retrieved_experiences": retrieved_experiences[:3],
        "prompt_preview": prompt_preview,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="学业规划最小可运行 Demo")
    parser.add_argument(
        "--input",
        type=str,
        default="我大二，GPA 3.4，没有科研，有一点竞赛，想保研，想知道现在该怎么规划。",
        help="用户输入的自然语言描述",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出完整结果",
    )
    args = parser.parse_args()

    result = run_demo(args.input)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("==== 用户画像 ====")
    print(json.dumps(result["user_profile"], ensure_ascii=False, indent=2))

    print("\n==== 检索查询 ====")
    print(result["user_profile"]["检索查询"])

    print("\n==== 相似学生 Top 3 ====")
    for item in result["top_matches"]:
        print(
            f"- {item.get('姓名', item['学生编号'])} | {item['目标']} | "
            f"GPA等级 {item['GPA等级']} | 科研 {item['科研强度']} | "
            f"竞赛 {item['竞赛强度']} | 相似度 {item['相似度']:.3f}"
        )

    print("\n==== 经验片段 Top 3 ====")
    for item in result["retrieved_experiences"]:
        print(
            f"- {item.get('姓名', item['学生编号'])} | {item['目标']} | "
            f"分数 {item['score']:.3f}\n  {item['chunk_text']}"
        )

    print("\n==== 最终建议 ====")
    print(result["answer"])


if __name__ == "__main__":
    main()
