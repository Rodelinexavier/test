import streamlit as st
import pickle
import pandas as pd
import numpy as np
import en_core_web_md
nlp = en_core_web_md.load()
from speech_recognition import Recognizer, Microphone
import datetime
import time
### Config
st.set_page_config(
    page_title="Web-detection",
    layout="wide"
)

### HEADER
st.title('Web Application to detect violents speeches')
st.header(" A  machine learning application to detect violent or non-violent speech")
st.markdown(""" This application allow to classify violent ou non-violent message.
 It attributes a score to every sentences. The purpose of this web application is to fight against 
 the hate speech at school and the social network.  We can anticipate cuberbullying, hate speech on the internet.
""")
st.title("How to use it")
st.markdown("* Click on the button` Detection violent speeches`.")
st.markdown ("* You must talk.After that, the app will register your message and transform it in a text-message.")
st.markdown("* It will give a hate score to every sentences and classify them ")

st.subheader('Detection violent speeches')

import streamlit as st
import speech_recognition as sr

def user_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit=10)
        try:
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            return e

with open('LR.pkl', 'rb') as LR:
    lr = pickle.load(LR)

with open('TFIDF.pkl', 'rb') as TFIDF:
    tfidf = pickle.load(TFIDF)

def hate_message_prob(msg):
    msg = tfidf.transform([msg])
    return LR.predict_proba(msg)[0][1]

def pred_prob_LR(text):
    token_text = nlp(text)
    text = [element.lemma_.lower() for element in token_text]
    text = " ".join(text)
    text_tfidf = tfidf.transform([text])
    probs = lr.predict_proba(text_tfidf)
    return probs[0][1]

st.sidebar.header("Time of the speech")
lt = time.localtime()
start_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)
start_date = st.sidebar.date_input('start date', start_datetime)
start_time = st.sidebar.text_input("start time", start_datetime.strftime("%H:%M"))

try:
    end_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour + 1, lt.tm_min)
except ValueError:
    try:
        end_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday + 1, 0, lt.tm_min)
    except ValueError:
        try:
            end_datetime = datetime.datetime(lt.tm_year, lt.tm_mon + 1, 1, 0, lt.tm_min)
        except ValueError:
            end_datetime = datetime.datetime(lt.tm_year + 1, 1, lt.tm_mday + 1, 0, lt.tm_min)

end_date = st.sidebar.date_input('end date', end_datetime)
end_time = st.sidebar.text_input("end time", end_datetime.strftime("%H:%M"))
start_date = start_date.strftime("%Y:%m:%d")
start_datetime = (start_date + ":" + start_time).split(":")
tuple_start_datetime = tuple([int(time) for time in start_datetime])
start_datetime = datetime.datetime(*tuple_start_datetime)
end_date = end_date.strftime("%Y:%m:%d")
end_datetime = (end_date + ":" + end_time).split(":")
tuple_end_datetime = tuple([int(time) for time in end_datetime])
end_datetime = datetime.datetime(*tuple_end_datetime)
lt = time.localtime()

current_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)
if st.button("Click me to begin: hate score recording will begin from starting time"):
    score_hate = []
    sentences = []
    st.write(f"It bigins at {start_datetime} and it ends at {end_datetime}")
    nb_sentences = 0
    while current_datetime <= end_datetime:
        lt = time.localtime()
        current_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)

        if current_datetime >= start_datetime:
            sentence = user_input()
            try:
                score_hate.append(hate_message_prob(sentence))
                sentences.append(sentence)
                nb_sentences += 1
                st.write(f"Sentence {nb_sentences}")
            except AttributeError:
                st.write("No sound detected")
    df = pd.DataFrame(columns=['Sentences', 'Rating of verbal violence'])
    df['Sentences'] = sentences
    df['Rating of verbal violence'] = score_hate
    score_hate_mean = round(np.mean(score_hate), 2)
    st.write(f'The hate score during this time is {score_hate_mean}')
    st.write(df)