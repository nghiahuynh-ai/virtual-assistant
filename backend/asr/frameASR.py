import torch
import numpy as np


class FrameASR:
    
    def __init__(
        self,
        model,
        sample_rate,
        window_stride,
        labels,
        data_layer,
        data_loader,
        chunk_size=1,
        frame_overlap=2.5, 
        offset=0,
    ):

        self.vocab = list(labels)
        self.vocab.append('_')
        
        self.model = model
        self.sr = sample_rate

        self.data_layer = data_layer
        self.data_loader = data_loader
        
    def _preprocess_audio(self, audio):
        if device is None:
            device = self.model.device()
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length
    
    @torch.no_grad()
    def transcribe(self, frame):
        (
            cache_last_channel, 
            cache_last_time, 
            cache_last_channel_len
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)

        chunk_audio, chunk_lengths = self._preprocess_audio(frame)

        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                asr_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
        )
        return transcribed_texts