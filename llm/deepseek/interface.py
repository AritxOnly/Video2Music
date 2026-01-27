from llm.interface import LLMInterface
from openai import OpenAI
import os

class DeepSeekInterface(LLMInterface):
    def __init__(self, model_name: str = 'deepseek-chat', api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')
        
        self.api_key = api_key
        self.model_name = model_name
        
        print(f"[DeepSeekInterface] Initializing client by api_key({"None" if not self.api_key else self.api_key[:10]})...")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """
        ActionGenerator 实际调用的方法，支持 System Prompt
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] API Error: {e}")
            raise e