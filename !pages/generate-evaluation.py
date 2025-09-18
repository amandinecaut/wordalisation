
# Library imports
import streamlit as st
import openai
from classes.data_source import PersonStat, CountryStats, PlayerStats
from classes.description import PersonDescription, CountryDescription, PlayerDescription
import copy
import json
import random
import pandas as pd
import utils.sentences as sentences
from tqdm import tqdm

import os
# from openai import OpenAI
from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE
def series_to_markdown(
    series, questions, header="| Factor | Z-score | Relevant question |"
):
    if questions is None:
        separator = "|:------|------:|"
        rows = [
            f"| {idx} | {val:.3f} |" for idx, val in zip(series.index, series.values)
        ]
    else:
        separator = "|:------|------:|:--------------------|"
        rows = [
            f"| {idx} | {val:.3f} | {q} |"
            for idx, val, q in zip(series.index, series.values, questions)
        ]
    return "\n".join([header, separator] + rows)

def generate_player_descriptions(n=10):
    players= PlayerStats()
    metrics = [m for m in players.df.columns if m not in ["player_name"]]
    players.calculate_statistics(metrics=metrics)
    player_names = players.df["player_name"].values.tolist()
    description_dict= dict(
        (
            metric,
            ["outstanding", "excellent", "good", "average", "below average", "poor"],
        )
        for metric in metrics
    )
    thresholds_dict = dict(
        (
            metric,
            [1.5, 1, 0.5, -0.5, -1],
        )
        for metric in metrics
    )
    def select_player(players, player_name):

        players = PlayerStats()
        players.calculate_statistics(metrics=metrics)
        # Make a copy of Players object
        player = copy.deepcopy(players)

        # rnd = int(player.select_random()) # does not work because of page refresh!
        # Filter player by position and select a player with sidebar selectors
        player.df = player.df[player.df["player_name"] == player_name]
        # Return data point

        player = player.to_data_point(gender="male", position="Forward")

        return player
    
    name_map = {
        "npxG_adjusted_per90": "non-penalty expected goals",
        "goals_adjusted_per90": "goals",
        "assists_adjusted_per90": "assists",
        "key_passes_adjusted_per90": "key passes",
        "smart_passes_adjusted_per90": "smart passes",
        "final_third_passes_adjusted_per90": "final third passes",
        "final_third_receptions_adjusted_per90": "final third receptions",
        "ground_duels_won_adjusted_per90": "ground duels",
        "air_duels_won_adjusted_per90": "air duels",
    }
    evaluation_descriptions=[]
    progress_bar = st.progress(0)
    for i, player_name in enumerate(tqdm(random.sample(player_names, n))):
        tmp_player = select_player(players, player_name)
        p_description = PlayerDescription(
            tmp_player,
        )
        data= p_description.player.ser_metrics
        # select only rows ending in "_Z"
        data = data[[col for col in data.index if col.endswith("_Z")]]
        data.index = [col[:-2] for col in data.index]
        data.index = [name_map.get(col, col) for col in data.index]
        questions = None
        numerical_parts = [f"{col}: {val:.2f}" for col, val in zip(data.index, data.values)]
        numerical_description = f"Here are the {player_name} z-scores on key football metrics: " + ", ".join(numerical_parts)
        synthetic_description=p_description.synthesize_text()
        result=stream_gpt("player",synthetic_description, numerical_description)
        # st.write(result)
        df_result = pd.DataFrame(result, columns=["Type", "LLMResponse"])
        # add player name column
        df_result["Name"] = player_name
        evaluation_descriptions.append(df_result)
        progress_bar.progress((i + 1) / n)
    progress_bar.empty()
    
    # combine all into one big DataFrame
    final_df = pd.concat(evaluation_descriptions , ignore_index=True)
    final_df.to_csv("evaluation/human-evaluation/data/player_descriptions.csv", index=False)

    
        

    
def generate_country_descriptions(n=10):
    country= CountryStats()
    metrics = [m for m in country.df.columns if m not in ["country"]]
    country.calculate_statistics(metrics=metrics)
    country_names = country.df["country"].values.tolist()

    with open("./data/wvs/description_dict.json", "r") as f:
        description_dict = json.load(f)
    thresholds_dict = dict(
            (
                metric,
                [
                    2,
                    1,
                    -1,
                    -2,
                ],
            )
            for metric in metrics
        )

    def select_country(country, country_name):

        country = CountryStats()
        country.calculate_statistics(metrics=metrics)
        country = copy.deepcopy(country)
        country.df = country.df[country.df["country"] == country_name]
        country = country.to_data_point()
        return country
    def get_country_questions(metric, c_description, entity):

        if metric.lower() in entity.drill_down_metrics:
            if entity.ser_metrics[metric + "_Z"] > 0:
                index = 1
            else:
                index = 0

            question, value = entity.drill_down_metrics[metric.lower()]
            question, value = question[index], value[index]
            description = "Question: '"
            description += c_description.relevant_questions[metric][question][0]
            description += "' Average answer:"
            description += c_description.relevant_questions[metric][question][1]
            description += " '"
            description += c_description.relevant_questions[metric][question][2][str(value)]
            description += "' "
            description += c_description.relevant_questions[metric][question][3]
            description += "."

        elif metric in entity.drill_down_metrics:

            if entity.ser_metrics[metric + "_Z"] > 0:
                index = 1
            else:
                index = 0

            question, value = entity.drill_down_metrics[metric]
            question, value = question[index], value[index]
            description = "Question: '"
            description += c_description.relevant_questions[metric][question][0]
            description += "' Average answer: "
            description += c_description.relevant_questions[metric][question][1]
            description += " '"
            description += c_description.relevant_questions[metric][question][2][str(value)]
            description += "' "
            description += c_description.relevant_questions[metric][question][3]
            description += "."
        else:
            description = ""
        return description
    evaluation_descriptions=[]

    progress_bar = st.progress(0)
    for i, country_id in enumerate(tqdm(random.sample(country_names, n))):
        # try:
        tmp_country = select_country(country, country_id)
        c_description = CountryDescription(
            tmp_country,
            description_dict=description_dict,
            thresholds_dict=thresholds_dict,
        )

        # text = f"```{c_description.synthesize_text()}```"
        data = c_description.country.ser_metrics
        # select only rows ending in "_Z"
        data = data[[col for col in data.index if col.endswith("_Z")]]
        data.index = [col[:-2] for col in data.index]
        questions = [
            get_country_questions(x,c_description, c_description.country) for x in data.index
        ]
        # Build the numerical description, adding questions after each data column if not empty
        numerical_parts = []
        for col, val, q in zip(data.index, data.values, questions):
            part = f"{col}: {val:.2f}"
            if q.strip():
                part += f" ({q})"
            numerical_parts.append(part)
        # st.write(numerical_parts)
        numerical_description = f"Here are the {country_id}'s z-scores on the World Values Survey: " + ", ".join(numerical_parts)
        synthetic_description=c_description.synthesize_text()
        # st.write(synthetic_description)
        # st.write(numerical_description) 
        result=stream_gpt("country",synthetic_description, numerical_description)
        # st.write(result)
        df_result = pd.DataFrame(result, columns=["Type", "LLMResponse"])
        # add country name column
        df_result["Name"] = country_id
        evaluation_descriptions.append(df_result)
        progress_bar.progress((i + 1) / n)
    progress_bar.empty()
    
    # combine all into one big DataFrame
    final_df = pd.concat(evaluation_descriptions , ignore_index=True)
    final_df.to_csv("evaluation/human-evaluation/data/country_descriptions.csv", index=False)

def generate_person_descriptions(n=10):
    people = PersonStat()

    metrics = [m for m in people.df.columns if m not in ["name"]]

    people.calculate_statistics(metrics=metrics)

    people_names = people.df["name"].values.tolist()
    def select_person(people, player_name):

        people = PersonStat()
        people.calculate_statistics(metrics=metrics)
        person = copy.deepcopy(people)
        person.df = person.df[person.df["name"] == player_name]
        person = person.to_data_point()

        return person

    def get_person_questions(metric, c_description, entity):

        questions = PersonStat().get_questions()
        description = " "

        if metric == "extraversion":
            if entity.ser_metrics[metric + "_Z"] > 1:
                index = entity.ser_metrics[0:10].idxmax()
                description = "In particular they said that " + questions[index][0] + "."
            if entity.ser_metrics[metric + "_Z"] < -1:
                index = entity.ser_metrics[0:10].idxmin()
                description = "In particular they said that " + questions[index][0] + "."
        elif metric == "neuroticism":
            if entity.ser_metrics[metric + "_Z"] > 1:
                index = entity.ser_metrics[10:20].idxmax()
                description = "In particular they said that " + questions[index][0] + ". "
            if entity.ser_metrics[metric + "_Z"] < -1:
                index = entity.ser_metrics[10:20].idxmin()
                description = "In particular they said that " + questions[index][0] + "."
        elif metric == "agreeableness":
            if entity.ser_metrics[metric + "_Z"] > 1:
                index = entity.ser_metrics[20:30].idxmax()
                description = "In particular they said that " + questions[index][0] + "."
            if entity.ser_metrics[metric + "_Z"] < -1:
                index = entity.ser_metrics[20:30].idxmin()
                description = "In particular they said that " + questions[index][0] + "."
        elif metric == "conscientiousness":
            if entity.ser_metrics[metric + "_Z"] > 1:
                index = entity.ser_metrics[30:40].idxmax()
                description = "In particular they said that " + questions[index][0] + "."
            if entity.ser_metrics[metric + "_Z"] < -1:
                index = entity.ser_metrics[30:40].idxmin()
                description = "In particular they said that " + questions[index][0] + "."
        elif metric == "openness":
            if entity.ser_metrics[metric + "_Z"] > 1:
                index = entity.ser_metrics[40:50].idxmax()
                description = "In particular they said that " + questions[index][0] + "."
            if entity.ser_metrics[metric + "_Z"] < -1:
                index = entity.ser_metrics[40:50].idxmin()
                description = "In particular they said that " + questions[index][0] + "."

        return description


    
    evaluation_descriptions=[]
    progress_bar = st.progress(0)
    for i, person_id in enumerate(tqdm(random.sample(people_names, n))):
        tmp_person = select_person(people, person_id)
        c_description = PersonDescription(
            tmp_person,
        )

        data = c_description.person.ser_metrics
        cols = [ 
            "extraversion_Z",
            "neuroticism_Z",
            "agreeableness_Z",
            "conscientiousness_Z",
            "openness_Z",
        ]
        data = data[[col for col in data.index if col in cols]]
        data.index = [col[:-2] for col in data.index]
        questions = [
            get_person_questions(x, c_description, c_description.person) for x in data.index
        ]   

        numerical_parts = []
        for col, val, q in zip(data.index, data.values, questions):
            part = f"{col}: {val:.2f}"
            if q.strip():
                part += f" ({q})"
            numerical_parts.append(part)
        numerical_description = "Here are the candidates z-scores on their personality test: " + ", ".join(numerical_parts)
        synthetic_description = c_description.synthesize_text()
        result = stream_gpt("person", synthetic_description, numerical_description)
        df_result = pd.DataFrame(result, columns=["Type", "LLMResponse"])
        df_result["Name"] = person_id
        evaluation_descriptions.append(df_result)
        progress_bar.progress((i + 1) / n)
    progress_bar.empty()
    
    # combine all into one big DataFrame
    final_df = pd.concat(evaluation_descriptions , ignore_index=True)
    final_df.to_csv("evaluation/human-evaluation/data/person_descriptions.csv", index=False)



def stream_gpt(entity, synthetic_description, numerical_description):
    # Set OpenAI API key
    openai.api_key = GPT_KEY
    # client = OpenAI()
    scaffolds={"W": json.load(open(f"evaluation/human-evaluation/prompts/wordalization_{entity}_prompt.json", encoding="utf-8")),
               "N": json.load(open(f"evaluation/human-evaluation/prompts/numerical_{entity}_prompt_v1.json", encoding="utf-8")),
               "Z": json.load(open(f"evaluation/human-evaluation/prompts/zero_knowledge_{entity}_prompt_v1.json", encoding="utf-8"))
               }

    PROMPT={
        "W": "Now do the same thing with the following: '''{synthetic_description}'''",
        "N": "Please describe the entity using the statistical information enclose with '''. Give a concise, 4 sentence summary. : '''{numerical_description}'''",
        "Z": "Please give a concise 4 sentence summary of the entity."

    }
    results=[]
    for key in scaffolds:
        prompt = PROMPT[key].format(
            synthetic_description=synthetic_description,
            numerical_description=numerical_description
        )

        messages = scaffolds[key] + [{"role": "user", "content": prompt}]
        openai.api_base = GPT_BASE
        openai.api_version = GPT_VERSION
        openai.api_key = GPT_KEY
        
        try:
            response= openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=messages,
                temperature=1,
            )
            llm_output = response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating {key} description: {e}")
            llm_output = "Error: " + str(e)
        finally:
            results.append((key, llm_output))
    return results

def combine_descriptions():
    person_desc = pd.read_csv("evaluation/human-evaluation/data/person_descriptions.csv")
    country_desc = pd.read_csv("evaluation/human-evaluation/data/country_descriptions.csv")
    player_desc = pd.read_csv("evaluation/human-evaluation/data/player_descriptions.csv")
    person_desc["entity"] = "person"
    country_desc["entity"] = "country"
    player_desc["entity"] = "player"
    combined = pd.concat([person_desc, country_desc, player_desc], ignore_index=True)
    combined.to_csv("evaluation/human-evaluation/data/all_descriptions.csv", index=False)
    st.write("### Description Counts")
    st.write(f"Person descriptions: {len(person_desc)}")
    st.write(f"Country descriptions: {len(country_desc)}")
    st.write(f"Player descriptions: {len(player_desc)}")
    st.write(f"##### {len(combined)} descriptions in total saved to evaluation/human-evaluation/data/all_descriptions.csv ")

from utils.page_components import add_common_page_elements
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()
# st.set_page_config(layout="wide")
st.title("Generate Evaluation Data")
num_persons = st.number_input("How many persons do you want to generate descriptions for?", min_value=1, value=10)
if st.button("Generate Person Descriptions"):
    generate_person_descriptions(num_persons)
num_countries = st.number_input("How many countries do you want to generate descriptions for?", min_value=1, value=10)
if st.button("Generate Country Descriptions"):
    generate_country_descriptions(num_countries)
num_players = st.number_input("How many players do you want to generate descriptions for?", min_value=1, value=10)
if st.button("Generate Player Descriptions"):
    generate_player_descriptions(num_players)

st.subheader("Combine All Descriptions")
if st.button("Combine All Descriptions"):
    combine_descriptions()