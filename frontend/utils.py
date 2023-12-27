import requests
import numpy as np


def getTranscript(audio, server, endpoint='asr'):
    if audio is None:
        return None
    response = requests.post(
        url=f'{server}/{endpoint}',
        files={'audio_file': audio}
    ).json()
    try:
        transcript = response['transcript']
    except:
        transcript = None
    return transcript


def genResponse(content, server, endpoint='llm'):
    response = requests.post(
        url=f'{server}/{endpoint}',
        json={'content': content}
    )
    return response