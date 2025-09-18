# ------------------------------
# Imports & Config
# ------------------------------
import random
import streamlit as st
import pandas as pd
import time
import uuid
import copy

from streamlit_gsheets import GSheetsConnection

from classes.data_source import PlayerStats, PersonStat, CountryStats
from classes.data_point import Player
from classes.visual import DistributionPlot, RadarPlot
from utils.page_components import add_common_page_elements
import streamlit.components.v1 as components

# st.set_page_config(layout="wide")

add_common_page_elements()  # your common sidebar/header


# ------------------------------
# Helper Functions
# ------------------------------
def select_person(player_name, metrics):
    people = PersonStat()
    people.calculate_statistics(metrics=metrics)
    person = copy.deepcopy(people)
    person.df = person.df[person.df["name"] == player_name]
    return person.to_data_point()


def select_player(player_name, metrics):
    players = PlayerStats()
    players.calculate_statistics(metrics=metrics)
    player = copy.deepcopy(players)
    player.df = player.df[player.df["player_name"] == player_name]
    return player.to_data_point(gender="male", position="Forward")


def select_country(country_name, metrics):
    countries = CountryStats()
    countries.calculate_statistics(metrics=metrics)
    country = copy.deepcopy(countries)
    country.df = country.df[country.df["country"] == country_name]
    return country.to_data_point()


def show_entity_plots(entity_type, entity_name, metrics):
    if entity_type == "person":
        entity = select_person(entity_name, metrics)
        dataset = PersonStat()
        dataset.calculate_statistics(metrics=metrics)
    elif entity_type == "player":
        entity = select_player(entity_name, metrics)
        dataset = PlayerStats()
        dataset.calculate_statistics(metrics=metrics)
    else:  # country
        entity = select_country(entity_name, metrics)
        dataset = CountryStats()
        dataset.calculate_statistics(metrics=metrics)

    visual_distribution = DistributionPlot(dataset, entity, metrics)
    visual_radar = RadarPlot(entity, metrics)

    col1, col2 = st.columns(2)
    with col1: visual_radar.show()
    with col2: visual_distribution.show()


def vote_question(key, question, options, number=None):
    label = f"**{number}. {question}**" if number else f"**{question}**"
    if st.session_state.get(key) is None:
        st.session_state[key] = None
    st.pills(label=label, options=options,  key=key)


def reset_questions():
    for key in ["faithfulness", "engagement", "usefulness", "hallucination", "comment"]:
        if key in st.session_state:
            del st.session_state[key]
    

# ------------------------------
# Intro Page
# ------------------------------
def show_intro():
    st.title("Welcome to the Evaluation Study 🎉")

    st.markdown("""
    Thank you for taking the time to participate in our study!  

    In this evaluation, you will:
    - Be shown a **plot** for a single entity (person, country, or player).  
    - Be shown a **generated description** of that plot.  
    - Answer **four short questions**:  
        1. Faithfulness – Does the description match the plot?  
        2. Engagement – Is it engaging?  
        3. Usefulness – Is it helpful for understanding?  
        4. Hallucinations – Are there unsupported claims?  

    **How it works:**  
    - Each page shows **one entity** with its plot + description.  
    - Answer the four questions, then submit.  
    - You can evaluate as many as you like (we’d be grateful for at least one).  
    - You are free to exit anytime.  

    ⚠️ **Note:** We do not collect personal info, only anonymous session activity.
    """)

    st.button("Start Evaluation ✅", on_click=lambda: st.session_state.update(show_intro=False))


# ------------------------------
# Demographics Page
# ------------------------------
def show_demographics():
    if "rater_id" not in st.session_state:
        st.session_state.rater_id = str(uuid.uuid4())
    st.title("Before we start, a few questions about you")

    st.markdown("""
    We are interested in understanding how different people perceive the generated descriptions.  
    Your answers will help us analyze the results better.  

    **Please note:** All responses are anonymous and confidential.  
    We do not collect any personal information beyond what you provide here.  
    You can skip any question you prefer not to answer.  
    """)

    age_input = st.number_input("1. How old are you? (optional)", min_value=10, max_value=120, step=1, value=18)
    st.session_state.age = age_input if age_input else None

    st.session_state.gender = st.selectbox("2. What is your gender?", ["Male", "Female", "Prefer not to say"])

    st.session_state.occupation = st.text_input("3. What is your occupation?")

    st.session_state.location = st.text_input("4. Where are you located?")

    st.session_state.education = st.selectbox("5. What is your highest level of education?", ["High School", "Bachelor's", "Master's", "PhD", "Other", "Prefer not to say"])

    st.session_state.english = st.selectbox(
        "8. How would you rate your English proficiency?",
        ["Basic", "Intermediate", "Advanced", "Native"]
    )
    
    st.session_state.experience = st.selectbox(
        "7. Do you have experience with data analysis or statistics?",
        ["No experience", "Some experience", "Experienced", "Expert"]
    )
    st.session_state.familiarity_gpt = st.selectbox(
        "9. How familiar are you with AI language models like GPT-3 or GPT-4?",
        ["Not familiar", "Somewhat familiar", "Familiar", "Very familiar"]
    )
    st.session_state.usage_gpt = st.selectbox(
        "10. How often do you use AI language models like GPT-3 or GPT-4?",
        ["Never", "Rarely", "Sometimes", "Often", "Very often"]
    )
    if st.button("Save and Proceed to Evaluation ✅"):
        # Save demographic data
        demographics_df=pd.DataFrame([{
            "rater_id": st.session_state.rater_id,
            "age": st.session_state.age,
            "gender": st.session_state.gender,
            "occupation": st.session_state.occupation,
            "location": st.session_state.location,
            "education": st.session_state.education,
            "experience": st.session_state.experience,
            "english": st.session_state.english,
            "familiarity_gpt": st.session_state.familiarity_gpt,
            "usage_gpt": st.session_state.usage_gpt,
            "timestamp": pd.Timestamp.now().isoformat()
        }])
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(ttl=0, worksheet="Demographic_Info")
        updated_data = pd.concat([existing_data, demographics_df], ignore_index=True)
        conn.update(worksheet="Demographic_Info", data=updated_data)


        st.session_state.get_demographics = False
        st.session_state.show_evaluation = True
        st.show_intro = False
        st.rerun()
        # get_demographics
        


# ------------------------------
# Evaluation Page
# ------------------------------
def show_evaluation():
    st.set_page_config(layout="wide")
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False
    
    if st.session_state.scroll_to_top:
        components.html(
            "<script>try{window.parent.scrollTo({top:0,behavior:'smooth'});}catch(e){try{window.top.scrollTo(0,0);}catch(e){}}</script>",
            height=0,
        )
        st.session_state.scroll_to_top = False

    
    df = pd.read_csv("evaluation/human-evaluation/data/all_descriptions.csv")
    conn = st.connection("gsheets", type=GSheetsConnection)
    # add a get a different question button

    # Right-align the button using columns
    button_col = st.columns([7, 2])[1]
    with button_col:
        st.button(
            "Get a Different Question",
            on_click=lambda: st.session_state.update(
                current_entity=random.choice([item for item in list(df[['Name', 'entity']].itertuples(index=False, name=None)) if item not in st.session_state.seen])
            )
        )


    # Session state init
    if "seen" not in st.session_state:
        st.session_state.seen = set()
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    all_items = list(df[['Name', 'entity']].itertuples(index=False, name=None))
    remaining = [item for item in all_items if item not in st.session_state.seen]

    if not remaining:
        st.write("✅ You have completed all evaluations. Thank you!")
        st.stop()

    if "current_entity" not in st.session_state or st.session_state.current_entity is None:
        st.session_state.current_entity = random.choice(remaining)

    st.session_state.seen.add(st.session_state.current_entity)
    entity_name, entity_type = st.session_state.current_entity
    row = df[(df['Name'] == entity_name) & (df['entity'] == entity_type)].iloc[0]

    # Show reference plot
    st.subheader(f"Ground Truth Reference for {entity_name} ({entity_type})")
    if entity_type == "person":
        metrics = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
    elif entity_type == "player":
        metrics = [
            "npxG_adjusted_per90", "goals_adjusted_per90", "assists_adjusted_per90",
            "key_passes_adjusted_per90", "smart_passes_adjusted_per90",
            "final_third_passes_adjusted_per90", "final_third_receptions_adjusted_per90",
            "ground_duels_won_adjusted_per90", "air_duels_won_adjusted_per90",
        ]
    else:  # country
        metrics = [m for m in CountryStats().df.columns if m not in ["country"]]

    show_entity_plots(entity_type, entity_name, metrics)

    # Centered description + questions
    
    center_col = st.columns([2, 6, 2])[1]
    with center_col:
        st.subheader("Description")
        st.write(row.LLMResponse)

        st.subheader("Questions")
        vote_question("faithfulness", "Does the text accurately represent the plot?", 
                      ["Completely inaccurate", "Mostly inaccurate", "Mostly accurate", "Completely accurate"], 1)
        vote_question("engagement", "Is the text engaging?", 
                      ["Not engaging", "Somewhat engaging", "Engaging", "Very engaging"], 2)

        if entity_type == "person":
            usefulness_q = "How useful is this description for a hiring decision?"
        elif entity_type == "country":
            usefulness_q = "How useful is this description for understanding the world value?"
        else:
            usefulness_q = "How useful is this description for learning about a football player?"

        vote_question("usefulness", usefulness_q, 
                      ["Very unuseful", "Unuseful", "Useful", "Very useful"], 3)

        vote_question("hallucination", "Does the text contain hallucinations (unsupported claims)?", ["No", "Yes"], 4)

        

        if st.session_state.get("hallucination") == "Yes":
            st.session_state.comment = st.text_area(
                "**5. Please highlight hallucinated parts of the text (optional):**"
            )

        # Submit
        if st.button("Submit and Continue", disabled=st.session_state.get("submitting", False)):
            st.session_state.submitting = True
            with st.spinner("Submitting your response..."):
                required = ["faithfulness", "engagement", "usefulness", "hallucination"]
                missing = [f for f in required if st.session_state.get(f) is None]
                if missing:
                    st.error(f"❌ Please answer all required questions: {', '.join(missing)}")
                    st.session_state.submitting = False
                    st.stop()

                response_time = time.time() - st.session_state.start_time
                response_data = {
                    "rater_id": st.session_state.rater_id,
                    "entity": entity_type,
                    "entity_id": row.Name,
                    "faithfulness": st.session_state.faithfulness,
                    "engagement": st.session_state.engagement,
                    "usefulness": st.session_state.usefulness,
                    "hallucination": st.session_state.hallucination,
                    "comment": st.session_state.get("comment", ""),
                    "response_time_sec": round(response_time, 2),
                    "timestamp": pd.Timestamp.now().isoformat()
                }

                # Append new row (instead of full read+concat)
                existing = conn.read(ttl=0)
                update = pd.concat([existing, pd.DataFrame([response_data])], ignore_index=True)
                conn.update(worksheet="Sheet1", data=update)

                st.success("✅ Response submitted!")
                time.sleep(2)
            
                # Reset for next round
                reset_questions()
                st.session_state.current_entity = None
                st.session_state.start_time = time.time()
                st.session_state.submitting = False
                st.session_state.scroll_to_top = True
                st.rerun()
                


# ------------------------------
# Main
# ------------------------------
if st.session_state.get("show_intro", True):
    show_intro()
elif st.session_state.get("get_demographics", True):
    show_demographics()
else:
    show_evaluation()
