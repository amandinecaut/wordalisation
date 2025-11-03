# ------------------------------
# Imports & Config
# ------------------------------
import random
import streamlit as st
import pandas as pd
import time
import uuid
import copy
import json
from streamlit_gsheets import GSheetsConnection
from classes.explainer import CountryExplanationProvider, PersonExplanationProvider
from classes.data_source import PlayerStats, PersonStat, CountryStats
from classes.data_point import Player
from classes.visual import DistributionPlot
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
    players.calculate_statistics(metrics=list(metrics.keys()))
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
        explanation_provider = PersonExplanationProvider(dataset.get_questions())
        person_plot_labels={
            "extraversion": ("Solitary & Reserved", "Outgoing & Energetic"),
            "neuroticism": ("Resilient & Confident", "Sensitive & Nervous"),
            "agreeableness": ("Critical & Rational", "Friendly & Compassionate"),
            "conscientiousness": ("Extravagant & Careless", "Efficient & Organized"),
            "openness": ("Consistent & Cautious", "Inventive & Curious")

        }
        visual_distribution= DistributionPlot(dataset, entity, metrics, explanation_provider=explanation_provider, labels=person_plot_labels, selected_entity=entity_name)
        # visual_radar = RadarPlot(entity, metrics, explanation_provider=explanation_provider)
    elif entity_type == "player":
        entity = select_player(entity_name, metrics)
        dataset = PlayerStats()
        dataset.calculate_statistics(metrics=list(metrics.keys()))
        metrics = list(metrics.values())
        visual_distribution= DistributionPlot(dataset, entity, metrics, selected_entity=entity_name)
        # visual_radar = RadarPlot(entity, metrics)
    else:  # country
        entity = select_country(entity_name, metrics)
        dataset = CountryStats()
        dataset.calculate_statistics(metrics=metrics)
        with open("data/wvs/intermediate_data/relevant_questions.json", "r") as f:
            relevant_questions = json.load(f)
        explanation_provider = CountryExplanationProvider(relevant_questions, entity.drill_down_metrics)
        country_plot_labels={
            "Traditional vs Secular Values": ("Traditional", "Secular"),
            "Survival vs Self-expression Values": ("Survival", "Self-Expression"),
            "Neutrality": ("Passive", "Active"),
            "Fairness": ("Permissive", "Principled"),
            "Skepticism": ("Trusting", "Skeptical"),
            "Societal Tranquility": ("Anxious", "Secure"),
        }
        visual_distribution= DistributionPlot(dataset, entity, metrics, explanation_provider=explanation_provider, labels=country_plot_labels, selected_entity=entity_name)
        # visual_radar = RadarPlot(entity, metrics, explanation_provider=explanation_provider)
        
    

    center_col = st.columns([0.2, 10, 0.2])[1]
    with center_col:
        visual_distribution.show()


def vote_question(key, question, options, number=None):
    label = f"**{number}. {question}**" if number else f"**{question}**"
    if st.session_state.get(key) is None:
        st.session_state[key] = None
    st.pills(label=label, options=options,  key=key)


def reset_questions():
    for key in ["faithfulness", "engagement", "usefulness", "hallucination", "comment"]:
        if key in st.session_state:
            del st.session_state[key]
    
def get_balanced_item(df_descriptions, conn_tracking):
    tracking_df = conn_tracking.read(ttl=0, worksheet="sample_tracking")
    df= df_descriptions.merge(tracking_df, on=['Name', 'entity', 'Type'], how='left')
    df['num_ratings']= df['num_ratings'].fillna(0)
    min_per_type = df.groupby("Type")["num_ratings"].min()
    overall_min = min_per_type.min()
    types_with_overall_min = min_per_type[min_per_type == overall_min].index.tolist()
    least_rated_type = df[
        df["Type"].isin(types_with_overall_min)
        & (df["num_ratings"] == df["Type"].map(min_per_type))
    ]
    # check st.session_state.seen to avoid already seen items
    least_rated_type = least_rated_type[~least_rated_type.set_index(['Name', 'entity']).index.isin(st.session_state.seen)]
    selected_row = least_rated_type.sample(1).iloc[0]
    return selected_row

def update_tracking(conn_tracking, name, entity,type ):
    tracking_df= conn_tracking.read(ttl=0, worksheet="sample_tracking")
    # ensure necessary columns exist
    for col in ["Name", "entity", "Type", "num_ratings"]:
        if col not in tracking_df.columns:
            tracking_df[col] = pd.Series(dtype="int" if col == "num_ratings" else "object")

    # locate matching row
    mask = (tracking_df["Name"] == name) & (tracking_df["entity"] == entity) & (tracking_df["Type"] == type)

    if mask.any():
        # increment existing count (handle NaNs)
        tracking_df.loc[mask, "num_ratings"] = tracking_df.loc[mask, "num_ratings"].fillna(0).astype(int) + 1
    else:
        # append new tracking row
        new_row = {"Name": name, "entity": entity, "Type": type, "num_ratings": 1}
        tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
    # update the worksheet
    conn_tracking.update(worksheet="sample_tracking", data=tracking_df)
    
# ------------------------------
# Intro Page
# ------------------------------
def show_intro():
    st.title("Welcome 🎉")

    st.markdown("""
    Thank you for taking the time to participate in our study!  

    We are conducting a study to evaluate the quality of LLM-generated descriptions for numerical data. We would greatly appreciate your help in assessing these descriptions.         
    """)

    st.markdown("""
    **How it works:**  
    Each page shows one subject and a chart about it.
    The chart shows where the subject sits compared to others.
    Any extra text you might see on the right or by hovering over the dots, explains what makes this subject special.

    Below the chart, you’ll see a short description of the subject. Please read it and answer four quick questions about how well it matches the data being shown in the chart.
    
    Click Submit to move to the next subject.
    You can rate as many as you like (even just one). No pressure — leave anytime.

    ⚠️ **Note:** We do not collect personal info aside from the questions we ask about you.  
    Your responses are anonymous and confidential.
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

    st.markdown("""
    <style>
    div[data-baseweb="input"] > div {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
    conn_tracking= st.connection("gsheets", type=GSheetsConnection)
    # add a get a different question button

    # Right-align the button using columns
    button_col = st.columns([7, 2])[1]
    with button_col:
        def get_different_question():
            # st.session_state.selected_entity_arm = None
            all_items = list(df[['Name', 'entity']].itertuples(index=False, name=None))
            remaining = [item for item in all_items if item not in st.session_state.seen]
            if remaining:
                reset_questions()
                # st.session_state.current_entity = random.choice(remaining)
                selected_row = get_balanced_item(df, conn_tracking)
                st.session_state.current_entity = (selected_row['Name'], selected_row['entity'], selected_row['Type'], selected_row['LLMResponse'])

            else:
                st.warning("✅ You have completed all evaluations. Thank you!")
                st.stop()

        st.button(
            "Get a different question",
            on_click=get_different_question
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
        # st.session_state.current_entity = random.choice(remaining)
        selected_row = get_balanced_item(df, conn_tracking)
        st.session_state.current_entity = (selected_row['Name'], selected_row['entity'], selected_row['Type'], selected_row['LLMResponse'])
        

    # add only the (Name, entity) pair to the seen set
    name, entity = st.session_state.current_entity[:2]
    st.session_state.seen.add((name, entity))
    entity_name, entity_type, evaluation_arm, llm_response = st.session_state.current_entity

    # Show reference plot
    st.subheader(f"Data for person {entity_name}" if entity_type == "person" else f"Data for {entity_name} ({entity_type})")
    if entity_type == "person":
        metrics = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
    elif entity_type == "player":
        metrics = {
            "npxG_adjusted_per90": "non-penalty expected goals",
            "goals_adjusted_per90": "goals",
            "assists_adjusted_per90": "assists",
            "key_passes_adjusted_per90": "key passes",
            "smart_passes_adjusted_per90": "smart passes",
            "final_third_passes_adjusted_per90": "final third passes",
            "final_third_receptions_adjusted_per90": "final third reception",
            "ground_duels_won_adjusted_per90": "ground duels",
            "air_duels_won_adjusted_per90": "air duels",
        }
        
    else:  # country
        metrics = [m for m in CountryStats().df.columns if m not in ["country"]]

    show_entity_plots(entity_type, entity_name, metrics)

    # Centered description + questions
    
    center_col = st.columns([1, 9, 1])[1]
    with center_col:
        if entity_type == "person":
            st.subheader(f"Description text for job candidate {entity_name}:")
        elif entity_type == "country":
            st.subheader(f"Description text for {entity_name}:")
        else:
            st.subheader(f"Description text for football player {entity_name}:")
        
        st.write(llm_response)

        st.subheader("Questions")
        vote_question("faithfulness", "Does the text accurately represent the plot?", 
                      ["Completely inaccurate", "Mostly inaccurate", "Mostly accurate", "Completely accurate"], 1)
        vote_question("engagement", "Is the text engaging?", 
                      ["Not engaging", "Somewhat engaging", "Engaging", "Very engaging"], 2)

        if entity_type == "person":
            usefulness_q = "How useful is this description for making a hiring decision?"
        elif entity_type == "country":
            usefulness_q = "How useful is this description for understanding the country's value system?"
        else:
            usefulness_q = "How useful is this description for scouting players?"

        vote_question("usefulness", usefulness_q, 
                      ["Not useful", "Partially useful", "Useful", "Very useful"], 3)

        vote_question("hallucination", "Does the text contain hallucinations (claims not supported by the figures above)?", ["No", "Yes"], 4)

        

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
                    "entity_id": entity_name,
                    "description_arm": evaluation_arm,
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
                update_tracking(conn_tracking, entity_name, entity_type, evaluation_arm)
                st.success("✅ Response submitted!")
                # time.sleep(2)
            
                # Reset for next round
                reset_questions()
                st.session_state.current_entity = None
                # st.session_state.selected_entity_arm = None
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
