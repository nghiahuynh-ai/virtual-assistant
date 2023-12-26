import time
import requests
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_chat import message
from audio_recorder_streamlit import audio_recorder


asr_backend = 'http://192.168.10.60:7070'


st.session_state.setdefault('audio_status', False)
st.session_state.setdefault('user', ['Hi there!'])
st.session_state.setdefault('bot', ['Please ask me anything...'])


def on_btn_click():
    del st.session_state.user[:]
    del st.session_state.bot[:]
    
    
def on_after_record(audio):
    if audio is None:
        return None, None, None, None, None, None

    response = requests.post(
        url=asr_backend,
        files={'audio_file': audio}
    ).json()
    try:
        transcript = response['transcript']
        asr_time = response['asr_time']
        audio = np.frombuffer(response['audio'], dtype=np.float32)
        audio_dtype = response['audio_dtype']
        audio_sr = response['audio_sr']
        tts_time = response['tts_time']
    except:
        transcript = None
        asr_time = None
        audio = None
        audio_dtype = None
        audio_sr = None
        tts_time = None
    return transcript, asr_time, audio, audio_dtype, audio_sr, tts_time


logo, title = st.columns(2, gap="small")
with logo:
    logo_img = Image.open("innotech-logo.png")
    st.image(logo_img, channels="BGR", width=250)
with title:
    st.write(
        "<h1 style='text-align: right; color: #F35D2D; font-size: 50px'>Virtual Agent</h1>",
        unsafe_allow_html=True
    )

chat_placeholder = st.empty()
clearButtonSpace, recordButtonSpace = st.columns([13.5, 1], gap="small")
audio_placeholder = st.empty()

with chat_placeholder.container():
    id = str(time.time())
    for i in range(len(st.session_state['user'])):
        message(st.session_state['user'][i], is_user=True, key=f'user_{id}_{i}')
        message(
            st.session_state['bot'][i], 
            key=f'bot_{id}_{i}', 
            allow_html=True,
            is_table=False
        )
          
with clearButtonSpace:
    st.write('\n')
    st.button("Clear message", on_click=on_btn_click)
        
with recordButtonSpace:
    st.write('\n')
    audio_bytes = audio_recorder(text='', pause_threshold=1000000, sample_rate=16000, icon_size='2x')
    transcript, asr_time, audio, audio_dtype, audio_sr, tts_time = on_after_record(audio_bytes)
        
    if transcript:
        st.session_state.user.append(transcript)
        st.session_state.bot.append(transcript)
        id = str(time.time())
        with chat_placeholder.container():
            for i in range(len(st.session_state['user'])):
                message(st.session_state['user'][i], is_user=True, key=f'user_{id}_{i}')
                message(
                    st.session_state['bot'][i], 
                    key=f'bot_{id}_{i}', 
                    allow_html=True,
                    is_table=False
                )

        audio = np.frombuffer(audio, dtype=audio_dtype)
        audio_placeholder.audio(audio)
