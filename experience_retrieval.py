import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from privacy_utils import anonymize_label, clean_experience_text


class ExperienceRetriever:
    def __init__(
        self,
        processed_data_path: str = "processed_students.csv",
        store_dir: str = "vector_store",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.df = pd.read_csv(processed_data_path).fillna("")
        self.store_dir = Path(store_dir)
        self.model_name = model_name
        self.use_vector_search = False
        self.index = None
        self.model = None
        self.chunk_metadata: List[Dict] = []
        self.chunks: List[str] = []

        self._load_chunks_from_dataframe()
        self._try_load_vector_store()

        print(f"已加载 {len(self.df)} 条学生记录，{len(self.chunks)} 个文本块。")
        print("当前使用向量检索模式。" if self.use_vector_search else "当前使用关键词检索模式。")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query.strip():
            return []
        if self.use_vector_search:
            return self._retrieve_with_vectors(query, top_k)
        return self._retrieve_with_keywords(query, top_k)

    def _load_chunks_from_dataframe(self) -> None:
        for _, row in self.df.iterrows():
            for chunk in self._parse_chunks(row.get("chunks", "[]")):
                clean_chunk = clean_experience_text(chunk)
                if not clean_chunk.strip():
                    continue
                item = {
                    "学生编号": row.get("学生编号", ""),
                    "显示名称": anonymize_label({"学生编号": row.get("学生编号", "")}),
                    "年级": row.get("年级", ""),
                    "目标": row.get("目标", ""),
                    "GPA等级": int(row.get("GPA等级", 0) or 0),
                    "科研强度": int(row.get("科研强度", 0) or 0),
                    "竞赛强度": int(row.get("竞赛强度", 0) or 0),
                    "chunk_text": clean_chunk,
                }
                self.chunk_metadata.append(item)
                self.chunks.append(clean_chunk)

    def _try_load_vector_store(self) -> None:
        index_path = self.store_dir / "student_chunks.faiss"
        metadata_path = self.store_dir / "chunk_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.index = faiss.read_index(str(index_path))
            with metadata_path.open("r", encoding="utf-8") as file:
                raw_metadata = json.load(file)
            self.chunk_metadata = []
            self.chunks = []
            for item in raw_metadata:
                clean_chunk = clean_experience_text(item.get("chunk_text", ""))
                if not clean_chunk:
                    continue
                normalized = {
                    "学生编号": item.get("学生编号", ""),
                    "显示名称": anonymize_label(item),
                    "年级": item.get("年级", ""),
                    "目标": item.get("目标", ""),
                    "GPA等级": int(item.get("GPA等级", 0) or 0),
                    "科研强度": int(item.get("科研强度", 0) or 0),
                    "竞赛强度": int(item.get("竞赛强度", 0) or 0),
                    "chunk_text": clean_chunk,
                }
                self.chunk_metadata.append(normalized)
                self.chunks.append(clean_chunk)
            self.model = SentenceTransformer(self.model_name)
            self.use_vector_search = True
        except Exception as exc:
            print(f"向量库加载失败，回退到关键词检索: {exc}")
            self.use_vector_search = False

    def _retrieve_with_vectors(self, query: str, top_k: int) -> List[Dict]:
        query_embedding = self.model.encode([query])
        query_embedding = np.asarray(query_embedding, dtype="float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results: List[Dict] = []
        for rank, chunk_idx in enumerate(indices[0]):
            if chunk_idx < 0 or chunk_idx >= len(self.chunk_metadata):
                continue
            item = dict(self.chunk_metadata[chunk_idx])
            item["score"] = float(1 / (1 + distances[0][rank]))
            results.append(item)
        return results

    def _retrieve_with_keywords(self, query: str, top_k: int) -> List[Dict]:
        query_terms = self._tokenize(query)
        scored_items: List[Dict] = []

        for item in self.chunk_metadata:
            text_terms = self._tokenize(item["chunk_text"])
            score = 0.0 if not query_terms else len(query_terms & text_terms) / len(query_terms)
            candidate = dict(item)
            candidate["score"] = score
            scored_items.append(candidate)

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        return [item for item in scored_items[:top_k] if item["score"] > 0]

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_]+", str(text).lower()))

    def _parse_chunks(self, chunk_value: str) -> List[str]:
        try:
            return json.loads(chunk_value or "[]")
        except json.JSONDecodeError:
            return []


if __name__ == "__main__":
    retriever = ExperienceRetriever()
    sample_query = "如何准备保研相关竞赛和科研"
    for item in retriever.retrieve(sample_query, top_k=3):
        print(f"{item['显示名称']} | {item['目标']} | {item['score']:.4f}")
        print(item["chunk_text"])
