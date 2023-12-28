import json
import time
import numpy as np
from io import BytesIO
from flask import Flask
from flask_cors import CORS
from tts.e2eTTS import E2ETTS
from pydub import AudioSegment
from nlp.chatgptEngine import LLM
from flask import request, Response
from asr.offlineASR import OfflineASR


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/asr', methods=['POST'])
async def transcribe():
    file = request.files['audio_file'].read()
    file = BytesIO(file)
    audio = AudioSegment.from_file(file=file, format='wav')
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    start = time.time()
    transcript = asr(audio)
    end = time.time()
    print('asr time: ', round(end - start, 3))

    result = {'transcript': transcript}

    return json.dumps(result)


@app.route('/gen', methods=['POST'])
async def gen_response():

    query = request.json['query']
    start = time.time()
    responseInText = llm(query)
    end = time.time()
    print('llm time: ', round(end - start, 3))

    start = time.time()
    audio = tts(responseInText)
    end = time.time()
    print('tts time: ', round(end - start, 3))
    
    responseInText = {
        'responseInText': responseInText,
        'responseInAudio': audio.tolist(),
        'audio_sr': sr,
    }
    return json.dumps(responseInText)


@app.route('/tts', methods=['POST'])
async def gen_voice():

    content = request.json['content']
    audio = tts(content)

    result = {
        'audio': audio.tolist(),
        'sr': sr,
    }

    return json.dumps(result)


if __name__ == '__main__':
    # http://127.0.0.1:7070
    sr = 22050
    asr = OfflineASR('stt_en_fastconformer_transducer_large', 'cpu')
    llm = LLM(stream=False)
    tts = E2ETTS('tts_en_fastpitch', 'tts_en_hifigan', sr, 'cpu')
    app.run(host='0.0.0.0', port=9090, debug=True)