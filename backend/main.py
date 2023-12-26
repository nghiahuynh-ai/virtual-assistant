import time
import json
import numpy as np
from io import BytesIO
from flask import Flask
from flask import request
from flask_cors import CORS
from pydub import AudioSegment
from asr.offlineASR import OfflineASR
from tts.e2eTTS import E2ETTS
from llm.llm import LLM
from omegaconf import OmegaConf


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/asr', methods=['POST'])
async def speech2text():
    file = request.files['audio_file'].read()
    file = BytesIO(file)
    audio = AudioSegment.from_file(file=file, format='wav')
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    transcript = asr(audio)
    return json.dumps({'transcript': transcript})


@app.route('/llm', methods=['POST'])
async def generate_response():
    return


@app.route('/tts', methods=['POST'])
async def text2speech():
    content = request.json()
    audio = tts(content)
    return json.dumps({
        'audio': audio.tobytes(),
        'sr': tts.sr,
    }) 


@app.route('/', methods=['POST'])
def get_data():
    file = request.files['audio_file'].read()
    file = BytesIO(file)
    audio = AudioSegment.from_file(file=file, format='wav')
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    transcript = asr(audio)

    audio = tts(transcript)

    result = {
        'transcript': transcript,
        'audio': audio.tolist(),
        'audio_sr': tts.sr,
    }
    return json.dumps(result)


if __name__ == '__main__':
    # http://127.0.0.1:7070
    config = OmegaConf.load('config.yaml')
    asr = OfflineASR(config)
    llm = LLM(config)
    tts = E2ETTS(config)
    app.run(host=config['host'], port=config['port'], debug=True)