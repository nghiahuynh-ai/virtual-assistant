import openai


class LLM:
    
    def __init__(self, config):
        
        openai.api_key = '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'
        
        self.model = config['chatgpt_name']
        self.prompt_base = config['prompt_base']
        
    def __call__(self, prompt):
        message = [
            {"role": "system", "content": self.prompt_base},
            {"role": "user", "content": prompt}
        ]
        return openai.ChatCompletion.create(
            model=self.model, 
            messages=message, 
            stream=True
        )