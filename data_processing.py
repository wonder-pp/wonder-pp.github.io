import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = {
    "姓名": ["姓名", "学生姓名", "名称"],
    "年级": ["年级", "当前阶段", "所在年级"],
    "绩点": ["绩点", "GPA", "绩点排名"],
    "科研经历": ["科研经历", "科研", "科研项目", "科研经验"],
    "竞赛": ["竞赛", "竞赛经历", "比赛经历", "学科竞赛"],
    "经验分享": [
        "经验分享",
        "保研经验分享",
        "考研经验分享",
        "就业经验分享",
        "留学经验分享",
        "创新创业类经验分享",
        "志愿经验分享",
        "国奖经验分享",
    ],
    "最终去向": ["最终去向", "最终出路", "录取院校", "目标", "去向"],
}


class DataProcessor:
    def __init__(self, input_path: str = "访谈.csv"):
        self.input_path = Path(input_path)

    def read_table(self, file_path: Optional[str] = None) -> pd.DataFrame:
        path = Path(file_path) if file_path else self.input_path
        readers = [
            lambda p: pd.read_csv(p, encoding="utf-8-sig"),
            lambda p: pd.read_csv(p, encoding="utf-8"),
            lambda p: pd.read_csv(p, encoding="gb18030"),
            pd.read_excel,
        ]

        last_error = None
        for reader in readers:
            try:
                df = reader(path)
                if not df.empty:
                    return df
            except Exception as exc:
                last_error = exc

        raise ValueError(f"无法读取数据文件: {path}. 最后一次错误: {last_error}")

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = pd.DataFrame()

        normalized["姓名"] = self._pick_series(df, DEFAULT_COLUMNS["姓名"], default_prefix="学生")
        normalized["年级"] = self._pick_series(df, DEFAULT_COLUMNS["年级"], default_value="")
        normalized["绩点"] = self._pick_series(df, DEFAULT_COLUMNS["绩点"], default_value="")
        normalized["科研经历"] = self._pick_series(df, DEFAULT_COLUMNS["科研经历"], default_value="")
        normalized["竞赛"] = self._pick_series(df, DEFAULT_COLUMNS["竞赛"], default_value="")
        normalized["最终去向"] = self._pick_series(df, DEFAULT_COLUMNS["最终去向"], default_value="")
        normalized["经验分享"] = self._merge_columns(df, DEFAULT_COLUMNS["经验分享"])

        normalized = normalized.fillna("")
        normalized["姓名"] = normalized["姓名"].astype(str).str.strip()
        normalized["姓名"] = normalized.apply(
            lambda row: row["姓名"] if row["姓名"] else f"学生{row.name + 1}",
            axis=1,
        )

        normalized["combined_text"] = (
            normalized["科研经历"].astype(str)
            + "\n"
            + normalized["竞赛"].astype(str)
            + "\n"
            + normalized["经验分享"].astype(str)
        ).str.strip()

        return normalized

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.replace("问：", "Q: ").replace("答：", "A: ")
        text = text.replace("问:", "Q: ").replace("答:", "A: ")
        return text.strip()

    def split_text(self, text: str, max_len: int = 120) -> List[str]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        parts = [part.strip() for part in re.split(r"[。！？!?；;\n]", cleaned) if part.strip()]
        chunks: List[str] = []
        current = ""

        for part in parts:
            if not current:
                current = part
                continue

            candidate = f"{current}。{part}"
            if len(candidate) <= max_len:
                current = candidate
            else:
                chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks

    def gpa_to_level(self, gpa_value: str) -> int:
        text = str(gpa_value).strip()
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match:
            gpa = float(match.group())
            if gpa >= 3.7:
                return 3
            if gpa >= 3.0:
                return 2
            return 1

        if any(keyword in text for keyword in ["优秀", "很高", "前10%", "前 10%"]):
            return 3
        if any(keyword in text for keyword in ["良好", "中上", "前30%", "前 30%"]):
            return 2
        if text:
            return 1
        return 2

    def research_level(self, text: str) -> int:
        text = str(text).strip()
        if any(keyword in text for keyword in ["论文", "EI", "SCI", "发表", "专利", "一作"]):
            return 3
        if any(keyword in text for keyword in ["大创", "项目", "课题", "实验室", "科研训练"]):
            return 2
        if len(text) > 5:
            return 1
        return 0

    def competition_level(self, text: str) -> int:
        text = str(text).strip()
        if any(keyword in text for keyword in ["国奖", "国家级", "一等奖", "全国"]):
            return 3
        if any(keyword in text for keyword in ["省级", "二等奖", "市级", "校级一等奖"]):
            return 2
        if len(text) > 5:
            return 1
        return 0

    def extract_goal(self, text: str) -> str:
        text = str(text)
        if any(keyword in text for keyword in ["保研", "推免", "直博"]):
            return "保研"
        if any(keyword in text for keyword in ["考研", "研究生", "硕士"]):
            return "考研"
        if any(keyword in text for keyword in ["留学", "出国", "海外", "雅思", "托福"]):
            return "留学"
        if any(keyword in text for keyword in ["就业", "工作", "求职", "公司"]):
            return "就业"
        if any(keyword in text for keyword in ["考公", "公务员", "选调"]):
            return "考公"
        return "未知"

    def process(self, file_path: Optional[str] = None) -> pd.DataFrame:
        raw_df = self.read_table(file_path)
        df = self.normalize_dataframe(raw_df)

        df["clean_text"] = df["combined_text"].apply(self.clean_text)
        df["chunks_list"] = df["clean_text"].apply(self.split_text)
        df["chunks"] = df["chunks_list"].apply(lambda items: json.dumps(items, ensure_ascii=False))
        df["GPA等级"] = df["绩点"].apply(self.gpa_to_level)
        df["科研强度"] = df["科研经历"].apply(self.research_level)
        df["竞赛强度"] = df["竞赛"].apply(self.competition_level)
        df["目标"] = df.apply(
            lambda row: self.extract_goal(f"{row['最终去向']} {row['经验分享']}"),
            axis=1,
        )
        df["学生编号"] = [f"S{i + 1}" for i in range(len(df))]

        return df[
            [
                "学生编号",
                "姓名",
                "年级",
                "绩点",
                "科研经历",
                "竞赛",
                "经验分享",
                "最终去向",
                "combined_text",
                "clean_text",
                "chunks",
                "GPA等级",
                "科研强度",
                "竞赛强度",
                "目标",
            ]
        ]

    def build_vector_store(
        self,
        processed_df: pd.DataFrame,
        store_dir: str = "vector_store",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> Tuple[Optional[object], List[str], List[Dict]]:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("未安装 sentence-transformers 或 faiss-cpu，跳过向量库构建。")
            return None, [], []

        model = SentenceTransformer(model_name)
        all_chunks: List[str] = []
        metadata: List[Dict] = []

        for _, row in processed_df.iterrows():
            chunk_items = self._parse_chunks(row.get("chunks", "[]"))
            for chunk in chunk_items:
                if not chunk.strip():
                    continue
                all_chunks.append(chunk)
                metadata.append(
                    {
                        "学生编号": row["学生编号"],
                        "姓名": row["姓名"],
                        "年级": row["年级"],
                        "目标": row["目标"],
                        "GPA等级": int(row["GPA等级"]),
                        "科研强度": int(row["科研强度"]),
                        "竞赛强度": int(row["竞赛强度"]),
                        "chunk_text": chunk,
                    }
                )

        if not all_chunks:
            print("没有可用于向量化的文本块。")
            return None, [], []

        embeddings = model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        store_path = Path(store_dir)
        store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(store_path / "student_chunks.faiss"))
        with (store_path / "chunk_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        print(f"向量库大小: {index.ntotal}")
        return index, all_chunks, metadata

    def save_processed_data(self, processed_df: pd.DataFrame, output_file: str = "processed_students.csv") -> None:
        processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"处理结果已保存到: {output_file}")

    def run(
        self,
        input_file: Optional[str] = None,
        output_file: str = "processed_students.csv",
        store_dir: str = "vector_store",
    ) -> pd.DataFrame:
        processed_df = self.process(input_file)
        self.save_processed_data(processed_df, output_file)
        self.build_vector_store(processed_df, store_dir=store_dir)
        return processed_df

    def _pick_series(
        self,
        df: pd.DataFrame,
        candidates: List[str],
        default_value: str = "",
        default_prefix: Optional[str] = None,
    ) -> pd.Series:
        for column in candidates:
            if column in df.columns:
                return df[column].fillna("").astype(str)

        if default_prefix:
            return pd.Series([f"{default_prefix}{i + 1}" for i in range(len(df))], index=df.index)

        return pd.Series([default_value] * len(df), index=df.index, dtype="object")

    def _merge_columns(self, df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        matched_columns = [column for column in candidates if column in df.columns]
        if not matched_columns:
            return pd.Series([""] * len(df), index=df.index, dtype="object")

        merged = []
        for _, row in df[matched_columns].fillna("").astype(str).iterrows():
            values = [value.strip() for value in row.tolist() if value.strip() and value.strip().lower() != "nan"]
            merged.append("\n".join(values))

        return pd.Series(merged, index=df.index, dtype="object")

    def _parse_chunks(self, chunk_value: str) -> List[str]:
        if isinstance(chunk_value, list):
            return [str(item) for item in chunk_value]
        try:
            return json.loads(chunk_value or "[]")
        except json.JSONDecodeError:
            return []


if __name__ == "__main__":
    processor = DataProcessor("访谈.csv")
    result_df = processor.run()
    print(f"共处理 {len(result_df)} 条学生记录。")
