## 高校数智学院学业导航智能体（协同过滤 + LLM）

本项目围绕“用往届学生数据集匹配新用户，结合 LLM 生成个性化发展建议”目标，实现了：

- **核心算法层**：基于改进协同过滤的学生相似性匹配（缺失填补 + 共同特征加权相似度 + 用户-项目双维融合），对应岳希等(2020)思路；
- **数据层**：面向 Excel/CSV 的往届学生数据读取、量化与清洗；
- **LLM 智能增强层**：封装 ChatGLM / 文心一言调用示例，用于生成自然语言化建议；
- **交互层（Streamlit）**：简单网页表单 + 匹配结果展示 + 雷达图 + 规划清单下载。

---

### 一、环境依赖

```bash
pip install -r requirements.txt
```

若使用文心一言，请保证 Python 版本 ≥ 3.8。

---

### 二、数据准备

1. 在项目根目录放置往届学生数据，例如 `demo_historical.csv`，包含字段示例：
   - `学生编号`：唯一标识；
   - `性格`：如“外向/中性/内向”等；
   - `家庭` 或 `家庭背景`：如“困难/一般/良好/优渥”；
   - `GPA`、`综测`、`高考`（可选）；
   - `竞赛`：次数或量化得分；
   - `发展意向` 或 `最终出路`：如“保研/考研/留学/考公/就业”；
   - 其他文本字段（如真实发展路径描述）可自由扩展，用于 LLM 解释。

2. 量化规则在 `data_processing.py` 中定义，可按学院标准修改，例如：
   - 性格量化：外向=4，中性=2，内向=1；
   - 发展意向编码：保研=1，考研=2，留学=3，考公=4，就业=5。

---

### 三、核心算法层（对应交付物 1）

- 文件：`cf_recommender.py`
- 核心类：`ImprovedCFMatcher`
  - **改进1：缺失值填补**
    - 方法 `_impute_missing_iterative`：
      - 先用特征全局均值粗略填补；
      - 再基于当前矩阵计算用户余弦相似度，使用“相似用户加权平均”迭代修正缺失特征；
      - 对应论文中“利用邻域预测缓解数据稀疏与冷启动”的思路。
  - **改进2：加权相似度（共同特征权重）**
    - 方法 `_user_similarity_with_common_weight`：
      - 基于余弦相似度；
      - 额外乘以“共同非缺失特征比例”的幂次权重 \(common\_ratio^\alpha\)；
      - 当共同维度较少时自动降低相似度，缓解数据稀疏导致的伪高相似。
  - **改进3：用户-项目双维相似度融合**
    - 方法 `_item_similarity` + `match_top_k`：
      - 把特征维度视为“项目”，对特征矩阵转置后做 item-based 相似度；
      - 构造新学生在各特征上的隐含偏好向量 \(p\_new = x\_new \cdot S\_{item}\)；
      - 再计算历史学生与 \(p\_new\) 的余弦相似度，作为 item-based 相似度；
      - 最终综合：`sim_final = λ * user_sim + (1-λ) * item_sim`。

快速测试：

```bash
python cf_recommender.py
```

> 需要先准备 `demo_historical.csv`。

---

### 四、数据处理脚本（对应交付物 2）

- 文件：`data_processing.py`
- 主要函数：
  - `quantize_and_clean(path) -> (df_q, feature_cols)`：
    - 自动识别 CSV / Excel；
    - 完成性格、家庭背景、成绩水平、发展意向、竞赛等特征的量化；
    - 对数值特征做 3σ 截断，缓解极端异常值；
    - 返回适配 `ImprovedCFMatcher` 的特征列列表。
  - `build_new_student_feature(...)`：
    - 将前端表单输入（家庭/性格/成绩评估/发展意向/竞赛强度）转换为特征字典。

---

### 五、LLM 调用示例（对应交付物 3）

- 文件：`llm_client.py`
- 核心类：`LLMClient`，通过 `LLMConfig` 配置 provider：
  - `provider="chatglm"`：使用智谱AI ChatGLM；
  - `provider="ernie"`：使用百度文心一言。
- 环境变量：
  - ChatGLM：
    - `ZHIPU_API_KEY`
    - 可选：`ZHIPU_MODEL`（默认 `glm-4`）
  - 文心一言：
    - `ERNIE_API_KEY`
    - `ERNIE_API_SECRET`
    - 可选：`ERNIE_MODEL`（默认 `ernie-3.5`）

示例用法：

```python
from llm_client import build_default_llm

llm = build_default_llm("chatglm")  # 或 "ernie"
text = llm.generate_plan(
    new_student_profile={"家庭背景": "一般", "性格": "偏内向", "发展意向": "考研"},
    matches=[...],  # 来自 ImprovedCFMatcher 返回的 Top-K 结果
    user_question="我想考研但目前竞赛科研几乎为0，该如何规划？",
)
print(text)
```

---

### 六、Streamlit 前端（对应交付物 4）

- 文件：`app_streamlit.py`
- 运行：

```bash
streamlit run app_streamlit.py
```

功能：
- 左侧：上传/选择历史数据集；
- 右侧主区：
  - 用户填写：家庭背景、性格、成绩水平、发展意向、竞赛情况、自述；
  - 选择 LLM 提供方（可关闭 LLM）；
  - 点击“开始匹配与规划”：
    - 调用 `ImprovedCFMatcher` 输出 Top-K 相似学生及相似度分解；
    - 绘制“新学生 vs Top-1”特征雷达图；
    - 调用 ChatGLM/文心一言生成自然语言化“匹配原因 + 个性化发展建议”；
    - 生成 Markdown 版“发展规划清单”，支持本地下载。

---

### 七、运行步骤总结（对应交付物 5）

1. 安装依赖：`pip install -r requirements.txt`  
2. 准备往届学生数据 `demo_historical.csv`（或通过前端上传 Excel/CSV）；  
3. 本地验证算法：
   - `python cf_recommender.py` 查看 Top-K 匹配效果；  
4. 启动前端：
   - `streamlit run app_streamlit.py` 在浏览器交互填写表单、查看匹配与建议；  
5. 如需开启 LLM：
   - 在系统环境变量中配置对应的 API Key / Secret，然后在前端选择 ChatGLM 或文心一言即可。

