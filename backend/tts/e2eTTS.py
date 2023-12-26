from typing import Any
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder


class E2ETTS:

    def __init__(self, generator_name, vocoder_name, sr=22050):
        
        try:
            self.generator = SpectrogramGenerator.from_pretrained(generator_name).eval()
            self.vocoder = Vocoder.from_pretrained(vocoder_name).eval()
            self.sr = sr
        except:
            raise ValueError('Cannot initialize the generator or vocoder model with the given names')
        
    def __call__(self, text):
        tokens = self.generator.parse(text, normalize=True)
        spec = self.generator.generate_spectrogram(tokens=tokens)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spec)
        return audio.numpy()[0]