# Royal College Oral Practice App

This is a natural language processing app that offers communications practice with patients across various clinical scenarios. Upon finishing the clinical scenario, the app provides AI-generated feedback and AI-generated SOAP notes at the user's request. This app is powered by Streamlit, OpenAI Whisper API, Google text-to-speech API, and GPT.

![Screenshot of the app](https://raw.githubusercontent.com/tig3r66/royal-college-practice/main/example.png)

## Installation

1. Ensure that you have [Homebrew](https://brew.sh/), [Git](https://git-scm.com/downloads), and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Clone this repository and set your working directory to `osce-gpt` by typing in your command line/terminal:

```
git clone https://github.com/tig3r66/royal-college-practice.git
cd royal-college-practice
```

3. Create and activate the Conda environment by typing in your command line/terminal:

```bash
conda create -n practice_env python=3.8
conda activate practice_env
```

4. Install the dependencies  by typing in your command line/terminal:

```bash
brew install portaudio
pip install -r requirements.txt
```

5. Create a `.env` file and add your OpenAI API key as such (see `.env.example` for an example). You can get an API by following [these instructions](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key).

6. Run the Streamlit website by typing in your command line/terminal:

```bash
streamlit run app.py
```
