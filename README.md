# 学业规划智能助手

这是一个基于真实学生访谈数据的学业规划 Demo，当前已经具备：

- 模块1：数据清洗、标签抽取、分块、向量化、向量库存储
- 模块2：用户画像解析、检索查询生成、Prompt 拼装
- 最小 Demo：输入一句话，输出画像、相似案例、经验片段和建议

## 安装依赖

```bash
pip install -r requirements.txt
```

## 1. 处理原始数据

```bash
python data_processing.py
```

执行后会生成：

- `processed_students.csv`
- `vector_store/student_chunks.faiss`（已安装向量依赖时）
- `vector_store/chunk_metadata.json`（已安装向量依赖时）

## 2. 运行命令行 Demo

```bash
python demo_pipeline.py --input "我大二，GPA 3.4，没有科研，有一点竞赛，想保研，想知道现在该怎么规划。"
```

如果想看完整 JSON：

```bash
python demo_pipeline.py --json
```

## 3. 运行网页 Demo

```bash
streamlit run app_main.py
```

## 4. 可选 LLM 配置

如果希望 `answer_generation.py` 调用模型生成答案，可以在 `.env` 中配置：

```env
ARK_API_KEY=你的密钥
ARK_MODEL=你的模型名
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
```

未配置时，系统会自动使用本地 fallback 规则生成建议。

## 当前主要文件

- `data_processing.py`：模块1数据处理
- `user_profiling.py`：模块2用户画像
- `prompt_builder.py`：Prompt 构建
- `experience_retrieval.py`：经验检索
- `peer_matching.py`：相似学生匹配
- `answer_generation.py`：答案生成
- `demo_pipeline.py`：最小可运行 Demo
- `app_main.py`：Streamlit 界面
