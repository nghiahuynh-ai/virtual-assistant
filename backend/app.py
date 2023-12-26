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


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/', methods=['POST'])
def get_data():
    file = request.files['audio_file'].read()
    file = BytesIO(file)
    audio = AudioSegment.from_file(file=file, format='wav')
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    start_asr_timestamp = time.time()
    transcript = asr(audio)
    end_asr_timestamp = time.time()

    start_tts_timestamp = time.time()
    audio = tts(transcript)
    end_tts_timestamp = time.time()

    result = {
        'transcript': transcript,
        'asr_time': round(end_asr_timestamp - start_asr_timestamp, 3),
        'audio': audio.tobytes(),
        'audio_dtype': type(audio[0]),
        'audio_sr': tts.sr,
        'tts_time': round(end_tts_timestamp - start_tts_timestamp, 3),
    }

    return json.dumps(result)


if __name__ == '__main__':
    # http://127.0.0.1:7070
    asr = OfflineASR('stt_en_fastconformer_transducer_large')
    tts = E2ETTS('tts_en_fastpitch', 'tts_en_hifigan', 22050)
    app.run(host='0.0.0.0', port=7070, debug=True)