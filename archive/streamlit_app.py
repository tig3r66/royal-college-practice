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

    def generate_response(self, prompt):
        self.update_memory("user", prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model='gpt-4',
            messages=self.memory,
            temperature=0.5,
            top_p=1,)['choices'][0]['message']['content']
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

    def update_history(self, question, response, text):
        self.history.append(f'Examiner: {question}')
        self.history.append(f'Examiner: {response}')
        self.history.append(f'Me: {text}')
        update_session_history(f'Examiner: {question}')
        update_session_history(f'Examiner: {response}')
        update_session_history(f'Me: {text}')

    def main(self):
        st.write("**Case initialized. You may begin speaking now.** You can end the scenario by clicking the *stop* button.")
        stop_button = st.button('Stop', disabled=st.session_state.feedback_state, on_click=feedback)
        st.markdown("---")

        if st.session_state.feedback_state is False:
            if self.option == OPTIONS[0].lower():
                case_info = "A previously healthy 52 year-old male presents to the emergency room with GCS 15 and a positive Romberg."
                st.write(case_info)
                self.speak(case_info)
                time.sleep(2)

                QUESTIONS = [
                    "Where is the hemorrhage?",
                    # "What are the parts of the vermis?",
                    "What is the management?",
                    # "Given this CTA, what are the findings?"
                    ]
                ANSWERS = [
                    "Midline cerebellum, likely vermian",
                    # "Lingula, central lobule, culmen (primary (tentorial) fissure), declive, folium (horizontal (petrosal) fissure), tuber (prebiventral/prepyramidal (suboccipital) fissure), pyramid, uvula, and nodule.",
                    "Admit to ICU, BP control, vascular imaging, needs decompressive surgery.",
                    # "Suspicious for small avm. Supply from distal PICA. Venous drainage overlies the distal PICA. Travels rostral along midline and then anterior towards the tentorium."
                    ]

                while stop_button is False:
                    for i in range(len(QUESTIONS)):
                        if i == 0:
                            st.image('cases/case2-img1.png', width=250)
                        elif i == 2:
                            st.image('cases/case2-img2.png', width=400)
                        question = f'Examiner: {QUESTIONS[i]}'
                        st.write(question)
                        self.speak(QUESTIONS[i])
                        text = self.get_user_voice()
                        st.write(f'Me: {text}')
                        response = self.generate_response(f'{text}\n\nModel response: {ANSWERS[i]}')
                        st.write(f'Examiner: {response}')
                        self.speak(response)
                        if text:
                            self.update_history(question, response, text)
                        time.sleep(1)
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
                instructions = 'Based on the chat dialogue between me and the patient, please provide constructive feedback and criticism for me, NOT the patient. Comment on the medical accuracy of responses. Comment on things that were done well, areas for improvement, and other remarks as necessary. Do not make anything up.'
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
        "You are a neurosurgeon evaluator for the Royal College neurosurgery oral exam. Provide feedback to me after every response. Use the model answer alongside your own knowledge and speak as if you were talking to me. Do not ask follow-up questions or request additional information. Treat this as an exam and do not provide words of encouragement.",
        ]

    while option == "Select one":
        time.sleep(1)

    if option == OPTIONS[0]:
        prompt = instructions[0]

    st.write(f'You selected: {option.lower()}')
    patient = Exam(prompt, option.lower())
    patient.main()
