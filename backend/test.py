import os
import openai
 

def gen():

    openai.api_key = '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'
    
    prompt = "what is innotech"
    
    result = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream = True # Add this optional property.
    )

    return result

def run(result):
    sentences, cur_sentence = [], ''
    for chunk in result:
        token = chunk.choices[0].delta.get("content", "")
        cur_sentence += token
        if token == '.':
            yield cur_sentence
            sentences.append(cur_sentence)
            cur_sentence = ''
    yield '<end>'

for chunk in run(gen()):
    print(chunk)