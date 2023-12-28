import requests
import numpy as np
import json


def getTranscript(audio, backend, endpoint='asr'):
    if audio is None:
        return None, None

    response = requests.post(
        url=f'{backend}/{endpoint}',
        files={'audio_file': audio}
    ).json()
    
    try:
        transcript = response['transcript']
    except:
        transcript = None

    return transcript


def genResponse(query, backend, endpoint='gen'):
    if query is None:
        return None, None, None
    
    response = requests.post(
        url=f'{backend}/{endpoint}',
        json={'query': query}
    ).json()

    try:
        responseInText = response['responseInText']
        responseInAudio = np.array(response['responseInAudio'], dtype=np.float32)
        audio_sr = response['audio_sr']
    except:
        responseInText, responseInAudio, audio_sr = None, None, None

    return responseInText, responseInAudio, audio_sr