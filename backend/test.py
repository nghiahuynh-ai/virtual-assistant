import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import librosa
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

# mysound = AudioSegment.from_wav("dump_error.wav")
# mysound = mysound.set_channels(1)
# mysound.export("dump_mono.wav", format="wav")

# model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained('stt_en_fastconformer_transducer_large')
# print(model.transcribe(['dump.wav']))

# sound, sr = librosa.load('dump.wav', sr=16000)
# print(sound.shape)

generator = SpectrogramGenerator.from_pretrained('tts_en_fastpitch').eval()
vocoder = Vocoder.from_pretrained('tts_en_hifigan').eval()