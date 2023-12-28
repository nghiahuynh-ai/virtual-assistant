import json


def chunking_text_audio_generator(streaming_llm_response, tts, audio_sr):
    cur_sentence = ''
    for chunk in streaming_llm_response:
        token = chunk.choices[0].delta.get("content", "")
        cur_sentence += token
        if token == '.':
            audio = tts(cur_sentence).tolist()
            yield json.dumps({
                'responseInText': cur_sentence,
                'responseInAudio': audio,
                'audio_sr': audio_sr
            })
            cur_sentence = ''