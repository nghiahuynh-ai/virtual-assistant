import openai
from omegaconf import OmegaConf


class LLM:

    def __init__(self, stream=False):
        config = OmegaConf.load('nlp/config.yaml')
        openai.api_key = config['key']
        with open('nlp/itv.txt', 'r') as f:
            itv_info = ''
            for line in f:
                line = line.replace('\n', '')
                itv_info += line
        self.itv_info = itv_info
        self.model_name = config['model_name']
        self.base_prompt = config['base_prompt']
        self.stream = stream

    def __call__(self, prompt):
        message = [
            {"role": "system", "content": self.itv_info},
            {'role': 'user', 'content': self.base_prompt + prompt}
        ]
        response = openai.ChatCompletion.create(
            model=self.model_name, 
            messages=message,
        )
        return response["choices"][0]["message"]["content"]