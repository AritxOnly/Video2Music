from typing import Literal
from llm.deepseek.interface import DeepSeekInterface
from llm.interface import LLMInterface
import dotenv

LLMBuiltinName = Literal["deepseek-v3", "deepseek-r1"]

def get_llm(name: LLMBuiltinName, **kwargs) -> LLMInterface:
    """
    统一创建 LLM 实例的工厂。
    kwargs 直接传给对应实现的 __init__。
    """
    if name == "deepseek-v3":
        return DeepSeekInterface(**kwargs)
    if name == "deepseek-r1":
        return DeepSeekInterface(model_name='deepseek-reasoner', **kwargs)

    raise ValueError(f"Unknown LLM backend: {name}")