import torch
import nemo.collections.asr as nemo_asr


class OfflineASR:

    def __init__(self, model_name, device='cpu'):
        
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location=device).eval()
            self.device = device
            self.model.preprocessor.featurizer.dither = 0.0
            self.model.preprocessor.featurizer.pad_to = 0
        except:
            raise ValueError('Cannot initialize the ASR model with the given name')
        
    def __call__(self, audio):
        signal, signal_length = self._preprocess(audio)
        output = self._transcribe(signal, signal_length)
        return output

    def _preprocess(self, audio):
        audio_length = len(audio)
        audio = torch.tensor(audio).unsqueeze(0)
        audio_length = torch.tensor(audio_length).unsqueeze(0)
        processed_signal, processed_signal_length = self.model.preprocessor(
                input_signal=audio, 
                length=audio_length,
        )
        return processed_signal, processed_signal_length

    @torch.no_grad()
    def _transcribe(self, signal, signal_length):
        encoded, encoded_len = self.model.encoder(
            audio_signal=signal, 
            length=signal_length
        )
        best_hyp, _ = self.model.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=False,
            partial_hypotheses=None,
        )
        return best_hyp[0]