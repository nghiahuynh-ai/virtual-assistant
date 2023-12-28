from typing import Any
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder


class E2ETTS:

    def __init__(self, generator_name, vocoder_name, sr=22050, device='cpu'):
        
        try:
            self.generator = SpectrogramGenerator.from_pretrained(generator_name, map_location=device).eval()
            self.vocoder = Vocoder.from_pretrained(vocoder_name).to(device).eval()
            self.sr = sr
            self.device = device
        except:
            raise ValueError('Cannot initialize the generator or vocoder model with the given names')
        
    def __call__(self, text):
        tokens = self.generator.parse(text, normalize=True)
        spec = self.generator.generate_spectrogram(tokens=tokens)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spec)
        if self.device == 'cpu':
            audio = audio.detach().numpy()[0]
        else:
             audio = audio.detach().cpu().numpy()[0]
        return audio