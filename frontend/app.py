import io
import base64
import streamlit as st
from scipy.io.wavfile import write
from audio_recorder_streamlit import audio_recorder
from utils import (
    getTranscript,
    genResponse,
)


backend = 'http://192.168.122.86:9090'


st.session_state.setdefault('user', [])
st.session_state.setdefault('bot', [])
robotIcon = '🤖'
humanIcon = '🧑'


_, title, _ = st.columns([0.5, 2.75, 1], gap="small")
# with logo:
#     logo_img = Image.open("innotech-logo.png")
#     st.image(logo_img, channels="BGR", width=250)
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
        transcript = getTranscript(input_audio_bytes, backend=backend)
        requestIntext = transcript
        if transcript:
            st.session_state.user.append(transcript)
            adjustLastMessages(transcript)
    
    if requestIntext:
        response, output_audio, audio_sr = genResponse(requestIntext, backend=backend)
        
        if response:
            st.session_state.bot.append(response)
            output_audio_bytes = np_audio_to_bytesio(output_audio, audio_sr)
            autoplay_audio(output_audio_bytes)
    
    reRenderChatHistory()


if __name__ == '__main__':
    main()