from llm.interface import LLMInterface
from openai import OpenAI
import os

class DeepSeekInterface(LLMInterface):
    def __init__(self, model_name: str = 'deepseek-chat', api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')
        
        self.api_key = api_key
        self.model_name = model_name
        
    def generate(self, prompt):
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content