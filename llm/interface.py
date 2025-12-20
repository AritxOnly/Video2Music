from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str:
        ...
    
