import torch
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder


class E2ETTS:

    def __init__(self, config):
        
        self.generator = SpectrogramGenerator.from_pretrained(config['tts_model_name'][0]).to('cpu').eval()
        self.vocoder = Vocoder.from_pretrained(config['tts_model_name'][1]).to('cpu').eval()
        self.sr = config['sr']
    
    @torch.no_grad()
    def __call__(self, text):
        tokens = self.generator.parse(text, normalize=True)
        spec = self.generator.generate_spectrogram(tokens=tokens)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spec)
        return audio.numpy()[0]