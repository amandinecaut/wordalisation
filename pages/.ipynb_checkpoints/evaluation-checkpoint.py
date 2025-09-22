# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import matplotlib.pyplot as plt
from utils.utils import normalize_text

from classes.data_source import PlayerStats
from classes.data_point import Player
import time
import numpy as np
from classes.visual import DistributionPlot

from utils.page_components import add_common_page_elements


# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

# --- Simulated data (replace with your own dataset / entity stats) ---
entity_name = "C_0"
metric = "Stamina"
entity_value = 7.3
cohort = np.random.normal(loc=6.0, scale=1.0, size=100)  # cohort distribution
cohort_median = np.median(cohort)

# --- LLM output (this would be generated under different arms) ---
llm_output = """The candidate is highly outgoing and energetic, exhibiting a strong tendency to engage socially, often taking the initiative to start conversations. While they are friendly and compassionate, they also display sensitivity and nervousness, leading them to experience more negative emotions and anxiety at times.
The candidate is very efficient and organized, demonstrating careful attention to detail and a diligent approach to their tasks. They are relatively consistent and cautious in their actions but tend to be less open to new ideas and experiences, favoring familiar routines over novelty."""

# --- Survey state ---
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

st.title("Output Evaluation Demo")



# 2. Show the ground truth plot for raters to compare
st.subheader(f"Ground Truth Reference for {entity_name}")
# embed picture
st.image("data/ressources/img/eval-demo.png", caption="Ground Truth Distribution (placeholder image)", use_column_width=True)



# 1. Show the generated text
st.subheader("LLM Generated Description")
st.write(llm_output)

# 3. Ask evaluation questions
st.subheader("Evaluation Questions")

#----------------------------------------
#faithfulness = st.slider("How faithful is the text to the ground truth?", 1, 7, 4)
#clarity = st.slider("How clear/readable is the text?", 1, 7, 4)
#trust = st.slider("How trustworthy/useful is the text?", 1, 7, 4)

# Initialize start time
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

entity_name = "Candidate A"


# Helper function to create a voting question
def vote_question(key, question, options):
    st.write(question)
    cols = st.columns(len(options))
    if key not in st.session_state:
        st.session_state[key] = None
    for i, (col, label) in enumerate(zip(cols, options), start=1):
        if col.button(label, key=f"{key}_{i}"):
            st.session_state[key] = i

vote_question("vote1", "Does the generated text accurately represent the candidate as depicted in the plot?", 
              ["Completely inaccurate", "Mostly inaccurate", "Mostly accurate", "Completely accurate"])

vote_question("vote2", "Is the text engaging?", 
              ["Not engaging", "Somewhat engaging", "Engaging", "Very engaging"])
# if entity_name in personality_test:
vote_question("vote3", "How useful would this description be if you were making a hiring decision?", 
              ["Very unuseful", "Unuseful", "Useful", "Very useful"])

#elif entity_name  in football:
#vote_question("vote3", "How useful this description is to get information on a football player?", 
#              ["Very unuseful", "Unuseful", "Useful", "Very useful"])
#else entity_name in wvs:
#vote_question("vote3", "How useful would this description is to understand how the world value works?"), 
#              ["Very unuseful", "Unuseful", "Useful", "Very useful"])



if "hallucination" not in st.session_state:
    st.session_state.hallucination = None

if "comment" not in st.session_state:
    st.session_state.comment = ""

st.session_state.hallucination = st.radio("Does the text contain hallucinations (unsupported claims)?", ["No", "Yes"])
st.session_state.comment = st.text_area("Optional comments:")


# 4. Save response + response time

if st.button("Submit and Continue"):
    response_time = time.time() - st.session_state.start_time
    st.session_state.start_time = time.time()

    response_data = {
        "entity": entity_name,
        "vote1": st.session_state.vote1,
        "vote2": st.session_state.vote2,
        "vote3": st.session_state.vote3,
        "hallucination": st.session_state.hallucination,
        "comment": st.session_state.comment,
        "response_time_sec": round(response_time, 2),
    }

    st.success("Response submitted! ✅")
    st.write(response_data)

    st.subheader("Session State Debug:")
    st.json(st.session_state)

# send the evaluation
import smtplib
from email.mime.text import MIMEText

# Simulated survey result
#survey_result = {
#    "vote1": st.session_state.get("vote1", None),
#    "vote2": st.session_state.get("vote2", None),
#    "vote3": st.session_state.get("vote3", None),
#    "hallucination": st.session_state.get("hallucination", None),
#    "comment": st.session_state.get("comment", ""),
#}

#if st.button("Send Survey via Email"):
#    sender_email = "your_email@gmail.com"
#    receiver_email = "recipient@example.com"
#    password = "your_app_password"  # Use app password for Gmail

#    # Create email message
#    subject = "Survey Results"
#    body = f"Here are the survey results:\n\n{survey_result}"
#    msg = MIMEText(body)
#    msg["Subject"] = subject
#    msg["From"] = sender_email
#    msg["To"] = receiver_email

#    try:
#        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#            server.login(sender_email, password)
#            server.sendmail(sender_email, receiver_email, msg.as_string())
#        st.success("Survey results sent successfully! ✅")
#    except Exception as e:
#        st.error(f"Error sending email: {e}")
# 
