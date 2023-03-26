import streamlit as st
import openai
import whisper
import pyttsx3
from transformers import GPT2TokenizerFast
import speech_recognition as sr
import time
import json

import gtts
from playsound import playsound

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")

with open('cases.json') as f:
    CASES = json.load(f)


class Exam:

    def __init__(self, instructions, option):
        if os.path.exists("audio.wav"):
            os.remove("audio.wav")
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")

        self.option = option.strip().lower()
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

        # from cases json
        self.images = CASES[self.option.strip().lower()][0]['images']

    def transcribe(self, audio):
        try:
            with open("audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            return self.model.transcribe('audio.wav', language='en', fp16=False)['text']
        except:
            return None

    def generate_response(self, prompt, user='user', pop_latest=False):
        self.update_memory(user, prompt)
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
        st.write("**Clinical scenario initialized.** You can end the scenario by clicking the *stop* button.")
        stop_button = st.button('Stop', disabled=st.session_state.feedback_state, on_click=feedback)
        st.markdown("---")

        if st.session_state.feedback_state is False:
            with sr.Microphone() as source:
                source.pause_threshold = 2.5  # silence in seconds
                first_q = True
                while True:
                    if stop_button:
                        break
                    if first_q:
                        response = self.generate_response('Provide a brief history of the case. If necessary, include the image files within parentheses but do not describe them.')
                        self.show_image(response, self.images)
                        st.write(f"Examiner: {response}")
                        self.speak(response)
                        first_q = False
                    # examiner question
                    response = self.generate_response('If the examinee asks a question or asks for further information, please answer. Then, ask question(s) about the case or ask follow-up question(s). Do not confirm this request, just directly answer the examinee.', 'system', True)
                    self.show_image(response, self.images)
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

def create_prompt(cases, option):
    instructions = f"Instructions: You are an evaluator for a neurosurgery oral exam. Provide contextual information to the examinee as relevant, as the examinee does not have any information of the case beforehand. Speak as if you were talking the examinee. Treat this as an exam and do not provide words of encouragement. Provide hints if the examinee does not know the answer.\n\nContext:{cases[option.strip().lower()][0]['case_info']}" + "\nPlease write the image files in parentheses if you would like to use them in your questions. If you've already used an image, no need to include it in parentheses."
    images = cases[option.strip().lower()][0]['images']

    image_prompt = '\n\nImages:\n'
    for i in range(len(images)):
        image_prompt += cases[option.strip().lower()][0]['images'][i]
        image_prompt += ': '
        image_prompt += cases[option.strip().lower()][0]['img_descriptions'][i]
        image_prompt += '\n'

    return instructions + image_prompt


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
        ("Select one", *list(CASES.keys())),
        disabled=st.session_state.disabled,
        on_change=disable,
    )

    while option == "Select one":
        time.sleep(1)

    prompt = create_prompt(CASES, option)

    st.write(f'You selected: {option}')
    patient = Exam(prompt, option)
    patient.main()
