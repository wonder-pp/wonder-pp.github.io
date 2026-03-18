"""
LLM 智能增强层：封装 ChatGLM / 文心一言 的调用示例。

设计原则：
- 不在代码中写死 API Key，统一从环境变量读取；
- 如果环境变量缺失或请求失败，外层可回退为规则模板输出；
- 这里只给出最小可运行示例，具体模型/参数可按学院实际调整。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Mapping, Optional

import requests


@dataclass
class LLMConfig:
    provider: str  # "chatglm" or "ernie"
    api_key: str
    api_secret: Optional[str] = None  # 文心一言需要
    model_name: str = ""
    base_url: Optional[str] = None


class LLMClient:
    """
    统一封装两个主流中文大模型接口：
    - ChatGLM (智谱AI)
    - 文心一言 (百度ERNIE)
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    # ----------------- Public API -----------------
    def generate_plan(
        self,
        *,
        new_student_profile: Mapping[str, str],
        matches: List[Mapping[str, object]],
        user_question: Optional[str] = None,
    ) -> str:
        """
        输入：
        - new_student_profile: 新学生的结构化画像（字典形式）
        - matches: Top-K 相似学生的特征 / 路径 / 结果列表
        - user_question: 用户自述或追问

        输出：
        - LLM 生成的自然语言化“匹配原因 + 发展建议”
        """
        system_prompt = (
            "你是一名高校数智学院的学业生涯规划导师。"
            "请根据输入的新生画像和与其最相似的往届学生案例，"
            "用通俗、可执行的方式给出：1) 匹配原因；2) 学业与发展规划建议；3) 阶段性时间线。"
            "必须显式引用数据证据（如：相似度、共同特征、成功路径），"
            "不要编造数据。"
        )

        user_prompt = {
            "new_student_profile": dict(new_student_profile),
            "top_k_matches": list(matches),
            "user_question": user_question or "",
        }

        if self.cfg.provider == "chatglm":
            return self._call_chatglm(system_prompt, user_prompt)
        if self.cfg.provider == "ernie":
            return self._call_ernie(system_prompt, user_prompt)
        if self.cfg.provider == "doubao":
            return self._call_doubao(system_prompt, user_prompt)
        raise ValueError(f"不支持的 provider: {self.cfg.provider}")

    # ----------------- ChatGLM 示例 -----------------
    def _call_chatglm(self, system_prompt: str, user_payload: object) -> str:
        """
        ChatGLM (智谱AI) 调用示例。
        参考官方文档：https://open.bigmodel.cn/
        """
        api_key = self.cfg.api_key or os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            return "【LLM未配置】未检测到 ChatGLM API Key，请在环境变量 ZHIPU_API_KEY 中配置。"

        model = self.cfg.model_name or "glm-4"
        base_url = (
            self.cfg.base_url
            or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.4,
        }
        try:
            resp = requests.post(base_url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return f"【ChatGLM调用失败】{type(e).__name__}: {e}"

    # ----------------- 文心一言(ERNIE) 示例 -----------------
    def _get_ernie_access_token(self) -> Optional[str]:
        """
        根据官方说明，先用 API Key + Secret Key 换取 access_token。
        文档参考：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5
        """
        api_key = self.cfg.api_key or os.getenv("ERNIE_API_KEY", "")
        secret_key = self.cfg.api_secret or os.getenv("ERNIE_API_SECRET", "")
        if not api_key or not secret_key:
            return None

        url = (
            "https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json().get("access_token")
        except Exception:
            return None

    def _call_ernie(self, system_prompt: str, user_payload: object) -> str:
        """
        文心一言调用示例，使用 ERNIE-Bot-turbo 等模型。
        """
        token = self._get_ernie_access_token()
        if not token:
            return "【LLM未配置】未正确配置 ERNIE API Key/Secret，请在环境变量 ERNIE_API_KEY / ERNIE_API_SECRET 中配置。"

        model = self.cfg.model_name or "ernie-3.5"
        base_url = (
            self.cfg.base_url
            or f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}"
        )

        headers = {"Content-Type": "application/json"}
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ]
        }
        try:
            resp = requests.post(
                base_url, params={"access_token": token}, headers=headers, json=body, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", "")
        except Exception as e:  # noqa: BLE001
            return f"【ERNIE调用失败】{type(e).__name__}: {e}"

    # ----------------- Doubao (火山引擎) 示例 -----------------
    def _call_doubao(self, system_prompt: str, user_payload: object) -> str:
        """
        Doubao (火山引擎 ARK API) 调用示例。
        参考官方文档：https://console.volcengine.com/ark
        """
        api_key = self.cfg.api_key or os.getenv("ARK_API_KEY", "")
        if not api_key:
            return "【LLM未配置】未检测到 Doubao API Key，请在环境变量 ARK_API_KEY 中配置。"

        model = self.cfg.model_name or os.getenv("VOLC_MODEL_NAME", "deepseek-v3")
        base_url = (
            self.cfg.base_url
            or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.4,
        }
        try:
            resp = requests.post(base_url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return f"【Doubao调用失败】{type(e).__name__}: {e}"


def build_default_llm(provider: str) -> LLMClient:
    """
    帮助函数：根据 provider 构建默认 LLMClient。
    只要在环境变量中配置好对应的 Key，即可直接使用。
    """
    provider = provider.lower()
    if provider == "chatglm":
        cfg = LLMConfig(
            provider="chatglm",
            api_key=os.getenv("ZHIPU_API_KEY", ""),
            model_name=os.getenv("ZHIPU_MODEL", "glm-4"),
        )
    elif provider == "ernie":
        cfg = LLMConfig(
            provider="ernie",
            api_key=os.getenv("ERNIE_API_KEY", ""),
            api_secret=os.getenv("ERNIE_API_SECRET", ""),
            model_name=os.getenv("ERNIE_MODEL", "ernie-3.5"),
        )
    elif provider == "doubao":
        cfg = LLMConfig(
            provider="doubao",
            api_key=os.getenv("ARK_API_KEY", ""),
            model_name=os.getenv("VOLC_MODEL_NAME", "deepseek-v3"),
            base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"),
        )
    else:
        raise ValueError("provider 仅支持 'chatglm'、'ernie' 或 'doubao'")
    return LLMClient(cfg)


__all__ = ["LLMConfig", "LLMClient", "build_default_llm"]

