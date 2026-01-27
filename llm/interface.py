from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str:
        ...
    
    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        ...
