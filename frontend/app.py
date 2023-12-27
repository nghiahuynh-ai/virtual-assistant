# import time
# import requests
# import numpy as np
# from PIL import Image
# import streamlit as st
# from streamlit_chat import message
# from audio_recorder_streamlit import audio_recorder


# asr_backend = 'http://172.19.94.162:7070'


# st.session_state.setdefault('audio_status', False)
# st.session_state.setdefault('user', ['Hi there!'])
# st.session_state.setdefault('bot', ['Please ask me anything...'])


# def on_btn_click():
#     del st.session_state.user[:]
#     del st.session_state.bot[:]
    
    
# def on_after_record(audio):
#     if audio is None:
#         return None, None, None

#     response = requests.post(
#         url=asr_backend,
#         files={'audio_file': audio}
#     ).json()
#     try:
#         transcript = response['transcript']
#         audio = np.array(response['audio'])
#         audio_sr = response['audio_sr']
#     except:
#         transcript = None
#         audio = None
#         audio_sr = None
#     return transcript, audio, audio_sr


# logo, title = st.columns(2, gap="small")
# with logo:
#     logo_img = Image.open("innotech-logo.png")
#     st.image(logo_img, channels="BGR", width=250)
# with title:
#     st.write(
#         "<h1 style='text-align: right; color: #F35D2D; font-size: 50px'>Virtual Agent</h1>",
#         unsafe_allow_html=True
#     )

# chat_placeholder = st.empty()
# clearButtonSpace, recordButtonSpace = st.columns([13.5, 1], gap="small")
# audio_placeholder = st.empty()

# with chat_placeholder.container():
#     id = str(time.time())
#     for i in range(len(st.session_state['user'])):
#         message(st.session_state['user'][i], is_user=True, key=f'user_{id}_{i}')
#         message(
#             st.session_state['bot'][i], 
#             key=f'bot_{id}_{i}', 
#             allow_html=True,
#             is_table=False
#         )
          
# with clearButtonSpace:
#     st.write('\n')
#     st.button("Clear message", on_click=on_btn_click)
        
# with recordButtonSpace:
#     st.write('\n')
#     audio_bytes = audio_recorder(text='', pause_threshold=1000000, sample_rate=16000, icon_size='2x')
#     transcript, audio, audio_sr = on_after_record(audio_bytes)
        
#     if transcript:
#         _ = requests.post(
#             url=f'{asr_backend}/llm',
#             json={'content': transcript}
#         ).json()
        
#         st.session_state.user.append(transcript)
#         st.session_state.bot.append(transcript)
#         id = str(time.time())
#         with chat_placeholder.container():
#             for i in range(len(st.session_state['user'])):
#                 message(st.session_state['user'][i], is_user=True, key=f'user_{id}_{i}')
#                 message(
#                     st.session_state['bot'][i], 
#                     key=f'bot_{id}_{i}', 
#                     allow_html=True,
#                     is_table=False
#                 )

#         audio_placeholder.audio(audio, sample_rate=audio_sr)


import io
import base64
from PIL import Image
import streamlit as st
from scipy.io.wavfile import write
from audio_recorder_streamlit import audio_recorder
from utils import (
    getTranscript,
    genResponse,
)


backend = 'http://172.29.247.88:7070'


st.session_state.setdefault('user', [])
st.session_state.setdefault('bot', [])
robotIcon = '🤖'
humanIcon = '🧑'


_, title, _ = st.columns([0.25, 3, 1], gap="small")
with title:
    st.write(
        "<h1 style='text-align: right; color: #F35D2D; font-size: 50px'>Virtual Assistant</h1>",
        unsafe_allow_html=True
    )

chat_container = st.container(border=True)
with chat_container:
    chat_history = st.container()
    last_messages = st.empty()
    with chat_history:
        with st.chat_message(name='user', avatar=humanIcon):
            st.write('Hi there!')
        with st.chat_message(name='bot', avatar=robotIcon):
            st.write('Please ask me anything...')

_, recordButtonSpace, _ = st.columns([1.75, 1, 1], gap="small")


def np_audio_to_bytesio(np_audio, np_audio_sr):
    _bytes = bytes()
    byte_io = io.BytesIO(_bytes)
    write(byte_io, np_audio_sr, np_audio)
    bytes_audio = byte_io.read()
    return bytes_audio


def autoplay_audio(audio: str):
    audio_base64 = base64.b64encode(audio).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)


def adjustLastMessages(userRequset):
    with last_messages.container():
        with st.chat_message(name='user', avatar=humanIcon):
            st.write(userRequset)
        with st.chat_message(name='bot', avatar=robotIcon):
            st.write('...')


def reRenderChatHistory():
    with chat_history:
        for user_state, bot_state in zip(st.session_state.user, st.session_state.bot):
            with st.chat_message(name='user', avatar=humanIcon):
                st.write(user_state)
            with st.chat_message(name='bot', avatar=robotIcon):
                st.write(bot_state)
    last_messages.empty()


def main():
    with recordButtonSpace:
        st.write('\n')
        input_audio_bytes = audio_recorder(
            text='', 
            energy_threshold=0.1, 
            pause_threshold=10, 
            sample_rate=16000, 
            icon_size='2x',
            key='recorder1'
        )

    prompt = st.chat_input('Say somthing...')

    requestIntext = None
    if prompt not in ['Say somthing...', '', None]:
        requestIntext = prompt
        st.session_state.user.append(prompt)
        adjustLastMessages(prompt)

    elif input_audio_bytes:
        transcript = getTranscript(input_audio_bytes, server=backend)
        requestIntext = transcript
        if transcript:
            st.session_state.user.append(transcript)
            adjustLastMessages(transcript)

    if requestIntext:
        response = genResponse(requestIntext, server=backend)
        print(response)
        for chunk in response:
            print(
                chunk.choices[0].delta.get("content", ""),
                end = "",
                flush = True
            )
        exit()
        if response:
            st.session_state.bot.append(response)
            reRenderChatHistory()
            output_audio_bytes = np_audio_to_bytesio(output_audio, audio_sr)
            autoplay_audio(output_audio_bytes)


if __name__ == '__main__':
    main()