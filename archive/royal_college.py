import streamlit as st
import openai
import whisper
import pyttsx3
from transformers import GPT2TokenizerFast
import speech_recognition as sr
import time

import gtts
from playsound import playsound

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")

OPTIONS = ['Hemorrhage']


class Exam:

    def __init__(self, instructions, option):
        if os.path.exists("audio.wav"):
            os.remove("audio.wav")
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")

        self.option = option
        self.engine = pyttsx3.init()
        self.memory = [{"role": "system", "content": instructions}]
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokens = len(self.tokenizer(instructions)['input_ids'])
        self.max_tokens = 4000
        self.r = sr.Recognizer()
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source, duration=2.5)
        self.r.dynamic_energy_threshold = True
        self.model = whisper.load_model("base")
        self.history = []

    def transcribe(self, audio):
        try:
            with open("audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            return self.model.transcribe('audio.wav', language='en', fp16=False)['text']
        except:
            return None

    def generate_response(self, prompt, pop_latest=False):
        self.update_memory("user", prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model='gpt-4',
            messages=self.memory,
            temperature=0.5,
            top_p=1,)['choices'][0]['message']['content']
        if pop_latest:
            self.memory.pop()
        self.update_memory("assistant", response)
        return response

    def generate_response_stream(self, memory):
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model='gpt-4',
            messages=memory,
            temperature=0.5,
            top_p=1,
            stream=True)
        return response

    def speak(self, text):
        gtts.gTTS(text).save("output.mp3")
        playsound("output.mp3")

    def update_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
        self.tokens += len(self.tokenizer(content)['input_ids'])
        while self.tokens > self.max_tokens:
            popped = self.memory.pop(0)
            self.tokens -= len(self.tokenizer(popped['content'])['input_ids'])

    def get_user_voice(self):
        with sr.Microphone() as source:
            source.pause_threshold = 1.5  # silence in seconds
            audio = self.r.listen(source)
            return self.transcribe(audio)
    
    def show_image(self, response, imgs_path):
        for img_path in imgs_path:
            if img_path in response:
                st.image(img_path, width=400)

    def main(self):
        st.write("**Clinical scenario initialized. You may begin speaking now.** You can end the scenario by clicking the *stop* button.")
        stop_button = st.button('Stop', disabled=st.session_state.feedback_state, on_click=feedback)
        st.markdown("---")

        if st.session_state.feedback_state is False:
            with sr.Microphone() as source:
                source.pause_threshold = 1  # silence in seconds
                while True:
                    if stop_button:
                        break
                    # examiner question
                    response = self.generate_response('If I ask a question or ask for further information, please answer me. Otherwise, please ask me a question about the case or ask a follow-up question.')
                    self.show_image(response, ['cases/case2-img1.png', 'cases/case2-img2.png'])
                    st.write(f"Examiner: {response}")
                    self.speak(response)
                    self.history.append(f'Examiner: {response}')
                    update_session_history(f'Examiner: {response}')

                    # user input
                    audio = self.r.listen(source)
                    text = self.transcribe(audio)
                    if text:
                        st.write(f'Me: {text}')
                        self.history.append(f'Me: {text}')
                        update_session_history(f'Me: {text}')
        else:
            for i in st.session_state.history:
                st.write(i)

        st.markdown("---")
        st.write('*Case ended.* Thank you for practicing with us! If you would like to practice again, please reload the page.')

        # feedback
        st.write('If you would like feedback, please click the button below.')
        feedback_button = st.button('Get feedback', key='feedback')
        if feedback_button:
            if len(st.session_state.history) != 0:
                instructions = 'Based on the chat dialogue between me and the patient, please provide constructive feedback and criticism for the resident ("Me:"), NOT the examiner. Comment on the medical accuracy of responses. Comment on things that were done well, areas for improvement, and other remarks as necessary. Do not make anything up.'
                temp_mem = [{'role': 'user', 'content': '\n'.join(st.session_state.history) + instructions}]
                stream = self.generate_response_stream(temp_mem)
                t = st.empty()
                full_response = ''
                for word in stream:
                    try:
                        next_word = word['choices'][0]['delta']['content']
                        full_response += next_word
                        t.write(full_response)
                    except:
                        pass
                    time.sleep(0.001)
            else:
                st.write('No conversation to provide feedback on.')


def disable():
    st.session_state.disabled = True

def feedback():
    st.session_state.feedback_state = True

def update_session_history(prompt):
    st.session_state.history.append(prompt)
    tokens = len(TOKENIZER('\n'.join(st.session_state.history))['input_ids'])
    max_tokens = 8000
    while tokens > max_tokens:
        st.session_state.history.pop(0)


if __name__ == '__main__':
    st.title('Royal College Oral Exam Practice')
    st.caption('Powered by Whisper, GPT-4, and Google text-to-speech.')
    st.caption('By [Eddie Guo](https://tig3r66.github.io/)')

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False
    if 'feedback_state' not in st.session_state:
        st.session_state.feedback_state = False

    option = st.selectbox(
        "Which clinical scenario would you like to practice with?",
        ("Select one", OPTIONS[0]),
        disabled=st.session_state.disabled,
        on_change=disable,
    )

    instructions = [
        "You are an evaluator for a neurosurgery oral exam. The case is a previously healthy 52 year-old male presents to the emergency room with GCS 15 and a positive Romberg. Speak as if you were talking to me. Treat this as an exam and do not provide words of encouragement. Provide hints if I don't know the answer. The two images in this case are 'cases/case2-img1.png' and 'cases/case2-im2.png'. The first image is a CT scan of a hemorrhage in the midline cerebellar region in the vermis. The second is a CTA showing an AVM. Please write the image files in parentheses if you would like to use them in your questions. If you've already used an image, no need to include it in parentheses.",
        ]

    while option == "Select one":
        time.sleep(1)

    if option == OPTIONS[0]:
        prompt = instructions[0]

    st.write(f'You selected: {option.lower()}')
    patient = Exam(prompt, option.lower())
    patient.main()
