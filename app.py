import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pyarrow.parquet as pq
import gc
import os
import re
import json
import requests
import psutil
import plotly.express as px

from google import genai
# from google.colab import userdata

import streamlit as st



def show_memory(label):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[RAM] {label}: {memory_mb:.1f} MB", flush=True)
#-----------Dataset Loading-----------------------------------------------------
#establish paths for processed and clean data
# processed_path = "/content/drive/MyDrive/CS X456.02/Project Data/processed/"
# clean_path = "/content/drive/MyDrive/CS X456.02/Project Data/clean/"
processed_path = "data/processed/"
clean_path = "data/clean/"
hf_base_url = "https://huggingface.co/datasets/cdube001/steam-game-mvp-data/resolve/main/"

show_memory("After imports")
#Reading the steam store clean data

Steam_Spy_df = pd.read_parquet(clean_path+"steamspy_apps_clean.parquet")

show_memory("After SteamSpy")

# @st.cache_data
# def load_steam_store():
#     return pd.read_parquet(
#         hf_base_url + "steam_store_clean.parquet"
#     )

# Steam_Store_df = load_steam_store()
# show_memory("After Steam Store")

#--------------Gemini Model-----------------------------------------------------
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

#--------------Functions--------------------------------------------------------

def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

def download_file(url,filepath):
        with requests.get(url, stream=True) as response:
                response.raise_for_status()

                with open(filepath,"wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                    f.write(chunk)


@st.cache_resource
def load_embedding_data():

    about_embeddings_path = "/tmp/about_this_game_embeddings.npy"
    about_appids_path = "/tmp/about_this_game_appids.npy"

    short_embeddings_path = "/tmp/short_description_embeddings.npy"
    short_appids_path = "/tmp/short_description_appids.npy"

    # Download About This Game embeddings
    if not os.path.exists(about_embeddings_path):
        download_file(
            hf_base_url + "about_this_game_embeddings.npy",
            about_embeddings_path
        )

    # Download About This Game app IDs
    if not os.path.exists(about_appids_path):
        download_file(
            hf_base_url + "about_this_game_appids.npy",
            about_appids_path
        )

    # Download Short Description embeddings
    if not os.path.exists(short_embeddings_path):
        download_file(
            hf_base_url + "short_description_embeddings.npy",
            short_embeddings_path
        )
        
    # Download Short Description app IDs
    if not os.path.exists(short_appids_path):
        download_file(
            hf_base_url + "short_description_appids.npy",
            short_appids_path
        )

    return (
        about_embeddings_path,
        about_appids_path,
        short_embeddings_path,
        short_appids_path
    )
 


# Returning top 100 results based on semantic vector search
# Semantic similarity captures conceptual relationships but may place substantial weight on named entities,
# so community-generated tags were incorporated as an additional signal to emphasize gameplay characteristics.
def top100semanticsearch(input_text):
    
    # Create Query Embedding
    show_memory("Before BGE")
    text_model = load_embedding_model()
        
    query_embedding = text_model.encode(
        input_text,
        convert_to_numpy=True
    )
    show_memory("After BGE loaded")
    
    del text_model
    gc.collect()
    show_memory("After BGE deleted")
    
    # Get embedding file paths
    (
        about_embeddings_path,
        about_appids_path,
        short_embeddings_path,
        short_appids_path
    ) = load_embedding_data()
    
    # About This Game similarity

    about_appids = np.load(about_appids_path)

    about_embedding_matrix = np.load(about_embeddings_path, mmap_mode="r")
    
    show_memory("About embedding mapped")
    
    about_this_game_similarity = cosine_similarity(
        query_embedding.reshape(1, -1),
        about_embedding_matrix
    )[0]
    show_memory("After About similarity")
    
    del about_embedding_matrix
    gc.collect()
    
    show_memory("After About embedding deleted")
    
    combined_scores_df = pd.DataFrame({
        "appid": about_appids,
        "about_this_game_score": about_this_game_similarity,
    })
    
    # Release objects no longer needed
    del about_appids
    del about_this_game_similarity
    gc.collect()
    
    # Short Description similarity
    short_appids = np.load(short_appids_path)

    short_embedding_matrix = np.load(short_embeddings_path, mmap_mode="r")
    show_memory("Short embedding mapped")
    
    short_description_similarity = cosine_similarity(
        query_embedding.reshape(1, -1),
        short_embedding_matrix
    )[0]
    show_memory("After Short similarity")
    
    del short_embedding_matrix
    del query_embedding
    gc.collect()
    
    show_memory("After Short deleted")
    
    short_scores = pd.DataFrame({
        "appid": short_appids,
        "short_description_score": short_description_similarity,
    })
    
    del short_appids
    del short_description_similarity
    gc.collect()
    
    # Combine scores
    combined_scores_df = combined_scores_df.merge(
        short_scores,
        on="appid",
        how="inner"
    )

    del short_scores
    gc.collect()
    
    combined_scores_df["combined_score"] = (
        0.8 * combined_scores_df["about_this_game_score"] +
        0.2 * combined_scores_df["short_description_score"]
    )

    # Top 100
    top_100 = combined_scores_df.nlargest(
        100,
        "combined_score"
    )
    
    
    show_memory("Before Steam Store")
    Steam_Store_df = pd.read_parquet(hf_base_url + "steam_store_clean.parquet")
    show_memory("After Steam Store loaded")

    print("COLUMNS:")
    print(Steam_Store_df.columns.tolist())
    
    print("\nMEMORY BY COLUMN:")
    print(
        Steam_Store_df.memory_usage(deep=True)
        .sort_values(ascending=False)
    )
    # Merge with Steam Store information
    steam_results = top_100[
        [
            "appid",
            "about_this_game_score",
            "short_description_score",
            "combined_score"
        ]
    ].merge(
        Steam_Store_df,
        on="appid",
        how="left"
    )
    show_memory("After Steam Store merge")
    
    del Steam_Store_df
    del top_100
    del combined_scores_df
    gc.collect()

    show_memory("After Steam Store deleted")
    
    return steam_results


#Tag scoring
def tag_scoring_on_semanticsearch(result_df):

            # result_df = top100semanticsearch(input_text)

            #Loading Dataset containing One-Hot Encoding of Community Tags
            OneHot_Steam_tags_df = pd.read_parquet(clean_path+"steam_community_tags_clean(one-hot).parquet")

            top_tag_rows = result_df[["appid", "combined_score"]].merge(OneHot_Steam_tags_df, on="appid", how="left")

            del OneHot_Steam_tags_df
            gc.collect()

            tag_frequency = (
                top_tag_rows
                .drop(columns=["appid","combined_score"])
                .mean()
                .sort_values(ascending=False)
            )

            tag_frequency_df = (
                tag_frequency
                .rename("tag_frequency")
                .reset_index()
                .rename(columns={"index":"tag"})
            )

            tag_frequency = tag_frequency[tag_frequency > 0]

            vector_frequency = tag_frequency.values.reshape(1,-1)

            game_tag_matrix = (
                top_tag_rows
                .set_index("appid")
                .reindex(columns=tag_frequency.index)
                .fillna(0)
            )

            # display(game_tag_matrix.shape)
            # vector_frequency.shape


            tag_similarity = cosine_similarity(
                game_tag_matrix.values,
                vector_frequency
            ).flatten()

            #reappending appid to tag scores
            tag_scores = pd.Series(
                tag_similarity,
                index=game_tag_matrix.index,
                name="tag_similarity"
            )
            tag_scores = tag_scores.reset_index()
            return tag_scores


def combined_scoring(input_text, sim_weight, tag_weight):

            result_df = top100semanticsearch(input_text)
            tag_scores = tag_scoring_on_semanticsearch(result_df)

            #merge tag scores with tag scores and semantic similarity
            combined = result_df.merge(tag_scores[["appid","tag_similarity"]], on="appid",how="left")

            #Using MinMax Scaler library
            scaler = MinMaxScaler()

            combined[["similarity_norm", "tag_similarity_norm"]] = scaler.fit_transform(
                combined[["combined_score", "tag_similarity"]]
            )

            combined["final_score"] = (
                ((sim_weight/100) * combined["similarity_norm"]) +
                ((tag_weight/100) * combined["tag_similarity_norm"])
            )

            combined = combined.sort_values(
                "final_score",
                ascending=False
            )

            return combined



# Due to the limited resources for gpu use on google colab, created a gemini option
def generate_response_gemini(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={
                "system_instruction": """You are a video game recommendation assistant.
                Follow the user's instructions exactly.
                When retrieved information is provided, prioritize that information
                and do not use outside knowledge unless the instructions explicitly allow it.""",
    
                "response_mime_type": "application/json",
    
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "retrieved_similar_games": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "rank": {"type": "INTEGER"},
                                    "game": {"type": "STRING"},
                                    "score": {"type": "NUMBER"}
                                },
                                "required": ["rank", "game", "score"]
                            }
                        },
    
                        "common_community_highlighted_gameplay_characteristics": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "characteristic": {"type": "STRING"},
                                    "frequency": {"type": "NUMBER"},
                                    "supporting_games": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"}
                                    }
                                },
                                "required": [
                                    "characteristic",
                                    "frequency",
                                    "supporting_games"
                                ]
                            }
                        },
    
                        "frequent_steam_features": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "feature": {"type": "STRING"},
                                    "frequency": {"type": "NUMBER"}
                                },
                                "required": ["feature", "frequency"]
                            }
                        },
    
                        "potential_gameplay_features_worth_considering": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "recommendation": {"type": "STRING"},
                                    "details": {"type": "STRING"}
                                },
                                "required": [
                                    "recommendation",
                                    "details"
                                ]
                            }
                        }
                    },
    
                    "required": [
                        "retrieved_similar_games",
                        "common_community_highlighted_gameplay_characteristics",
                        "frequent_steam_features",
                        "potential_gameplay_features_worth_considering"
                    ]
                }
            }
        )
    
        return json.loads(response.text)

    except Exception as e:
            print(f"Gemini error: {e}")
            return None
def create_rag_context(df):

  #Initialzing the context that will create structure for the model to understand the prompt
  context =""

  for i, (_, game) in enumerate(df.iterrows(), start=1):
      # Passing each game in the dataset (top results only), providing the name, description, and community tags
      context += f"""
                    --- RETRIEVED GAME {i} -------
                    Game:
                    {game['name']}

                    Description:
                    {game['about_this_game']}

                    Community Tags:
                    {game['tag_names']}

                    """
  return context


def steam_feature_frequency(df):

    #Loading Dataset containing features and their groupings
    Steam_feature_df = pd.read_parquet(processed_path+"exploded_steam_features.parquet")

    retrieved_features = Steam_feature_df[Steam_feature_df['appid'].isin(df['appid'])]

    #ignoring Steam features since this is not a feature that is implemented by the developer
    excluded_groups = ['platform_features']

    retrieved_features = retrieved_features[~retrieved_features['feature_group'].isin(excluded_groups)]

    total_games = df['appid'].nunique()

    #counting the frequency of the features
    feature_freq = retrieved_features.groupby(['feature_group','categories'])['appid'].nunique().sort_values(ascending=False).reset_index(name='game_count')


    # Calculate percentage
    feature_freq['percentage'] = (
        feature_freq['game_count'] / total_games
    )


    #if the feature is not in at least 2 games, remove
    feature_freq = feature_freq[feature_freq['game_count'] >= 2]

    #Conserving Space
    del Steam_feature_df
    gc.collect()

    return feature_freq

def steam_feature_group_frequency(df):
    feature_frequency = steam_feature_frequency(df)

    group_frequency = (
        feature_frequency
        .groupby('feature_group')['game_count']
        .max()
        .reset_index(name='game_count')
    )

    total_games = df['appid'].nunique()

    group_frequency['Percentage'] = (
        group_frequency['game_count'] / total_games
    )

    return group_frequency


def steam_feature_frequency_context(feature_frequency, total_games):
    #formatting the frequency results into a string to pass into the model
    context = "Features and their frequency among similar games:\n"

    for _, feature in feature_frequency.iterrows():

        #Formating the steam feature with its name, the group of features it belongs to, how often it occurs in the results,
        # and the percentage representation of that result
        context += (
            f" - {feature['categories']} "
            f"({feature['feature_group']}): "
            f"{feature['game_count']}/{total_games} games"
            f" ({feature['percentage']:.0%})\n"
        )
    return context



def community_tag_frequency_context(df):

    #Breaking down the list to observe eaxh tag separately
    retrieved_tags = df[['appid','tag_names']].explode('tag_names')

    #Creating a list of tags and how often the occur
    tag_frequency = retrieved_tags.groupby('tag_names')['appid'].nunique().sort_values(ascending=False).reset_index(name='game_frequency')

    #Removing unique tags that only appear once
    tag_frequency = tag_frequency[tag_frequency['game_frequency'] >= 2]

    #Formatting the results into a string to pass into the model
    tag_context = "Common Community-Highlighted Gameplay Characteristics:\n\n"

    #Calculating the total amount of games being evaluated
    total_games = df['appid'].nunique()

    # Calculating percentage
    tag_frequency['percentage'] = (
        tag_frequency['game_frequency'] / total_games
    )

    for _, tag in tag_frequency.iterrows():
        #Calculating the percentage of the tag appearing across the games


        #Formating the community tag with its name, how often it occurs in the results,
        # and the percentage representation of that result
        tag_context += (
            f" - {tag['tag_names']} "
            f" {tag['game_frequency']}/{total_games} games"
            f" ({tag['percentage']:.0%})\n"
        )
    return tag_context, tag_frequency


#---------------------TEST------------------------------------------------------
def test_gemini_response():

    return {
        "retrieved_similar_games": [
            {
                "rank": 1,
                "game": "DOOM",
                "score": 0.914
            },
            {
                "rank": 2,
                "game": "ULTRAKILL",
                "score": 0.887
            },
            {
                "rank": 3,
                "game": "Warhammer 40,000: Boltgun",
                "score": 0.852
            },
            {
                "rank": 4,
                "game": "Prodeus",
                "score": 0.826
            },
            {
                "rank": 5,
                "game": "Turbo Overkill",
                "score": 0.798
            }
        ],

        "common_community_highlighted_gameplay_characteristics": [
            {
                "characteristic": "First-Person Shooter",
                "frequency": 5,
                "supporting_games": [
                    "DOOM",
                    "ULTRAKILL",
                    "Warhammer 40,000: Boltgun",
                    "Prodeus",
                    "Turbo Overkill"
                ]
            },
            {
                "characteristic": "Fast-Paced Combat",
                "frequency": 4,
                "supporting_games": [
                    "DOOM",
                    "ULTRAKILL",
                    "Prodeus",
                    "Turbo Overkill"
                ]
            },
            {
                "characteristic": "Single-player",
                "frequency": 4,
                "supporting_games": [
                    "DOOM",
                    "ULTRAKILL",
                    "Boltgun",
                    "Prodeus"
                ]
            }
        ],

        "frequent_steam_features": [
            {
                "feature": "Steam Achievements",
                "frequency": 5
            },
            {
                "feature": "Steam Cloud",
                "frequency": 4
            },
            {
                "feature": "Controller Support",
                "frequency": 4
            }
        ],

        "potential_gameplay_features_worth_considering": [
            {
                "recommendation": "In-Game Mod & Map Browser",
                "details": "Consider providing an integrated browser for discovering, downloading, and sharing community-created maps."
            },
            {
                "recommendation": "Aggressive Combat-Driven Health Mechanics",
                "details": "Consider rewarding aggressive close-range combat with health, armor, or ammunition recovery to encourage continuous momentum."
            },
            {
                "recommendation": "Asymmetric Multiplayer Modes",
                "details": "Consider specialized multiplayer modes that give different players distinct abilities or roles."
            },
            {
                "recommendation": "Cross-Platform Multiplayer",
                "details": "Consider supporting cross-platform multiplayer to increase the potential player pool for cooperative and competitive modes."
            }
        ]
    }

def game_concept(input_text, similarity_weight, tag_weight, top_games):

    show_memory("Game concept START")

    combined_results = combined_scoring(
        input_text,
        similarity_weight,
        tag_weight
    )

    show_memory("After combined scoring")

    # Top 100 for analysis
    analysis_results = combined_results.head(100).copy()

    # Top games for RAG
    rag_results = combined_results.head(top_games)

    show_memory("After analysis/rag results")

    # ---------------------------------------------------------------------------
    # Top 100 Analysis
    # ---------------------------------------------------------------------------

    list_features = steam_feature_frequency(analysis_results)

    feature_group_results = steam_feature_group_frequency(
        analysis_results
    )

    feature_context = steam_feature_frequency_context(
        list_features,
        100
    )

    show_memory("After feature analysis")

    # Loading Community Tags
    community_Steam_tags_df = pd.read_parquet(
        clean_path + "steam_community_tags_clean.parquet",
        columns=["appid", "tag_names"]
    )
    show_memory("After community tags loaded")
    community_Steam_tags_df = community_Steam_tags_df[
        community_Steam_tags_df["appid"].isin(
            analysis_results["appid"]
        )
    ].copy()
    show_memory("After community tags filtered")
    

    # Merge community tags
    analysis_results = analysis_results.merge(
        community_Steam_tags_df[['appid', 'tag_names']],
        on="appid",
        how="left"
    )

    show_memory("After community tags merged")

    # SteamSpy popularity metrics
    analysis_results = analysis_results.merge(
        Steam_Spy_df[['appid', 'ccu', 'positive', 'negative']],
        on="appid",
        how="left"
    )

    show_memory("After SteamSpy merged")

    analysis_results['review_count'] = (
        analysis_results['positive'].fillna(0) +
        analysis_results['negative'].fillna(0)
    )

    analysis_results['log_ccu'] = np.log1p(
        analysis_results['ccu']
    )

    analysis_results['log_reviews'] = np.log1p(
        analysis_results['review_count']
    )

    scaler = MinMaxScaler()

    analysis_results[['CCU Score', 'Review Score']] = (
        scaler.fit_transform(
            analysis_results[['log_ccu', 'log_reviews']]
        )
    )

    analysis_results['Engagement Score'] = (
        0.4 * analysis_results['CCU Score'] +
        0.6 * analysis_results['Review Score']
    )

    popular_similar_games = (
        analysis_results
        .sort_values(
            'Engagement Score',
            ascending=False
        )
        .head(top_games)
        .copy()
    )

    show_memory("After popularity calculation")

    # Tag frequency
    tag_context, tag_frequency = (
        community_tag_frequency_context(
            analysis_results
        )
    )

    show_memory("After tag frequency")

    # st.write("Analysis Results:")
    # st.dataframe(analysis_results)

    # Market summary
    market_data = analysis_results[["appid",'initial_price','current_price','is_free','coming_soon','release_year','release_month','release_day']]
    
    show_memory("After market summary")

    # ---------------------------------------------------------------------------
    # RAG
    # ---------------------------------------------------------------------------

    rag_results = rag_results.merge(
        community_Steam_tags_df,
        on="appid",
        how="left"
    )

    show_memory("After RAG tag merge")

    del community_Steam_tags_df
    gc.collect()

    show_memory("After community tags deleted")

    game_context = create_rag_context(rag_results)

    show_memory("After RAG context")

    prompt = f"""

                  You are analyzing a proposed video game concept.

                  Use only the retrieved Steam game information provided below for Sections 1-3.

                  1. Retrieved Similar Games:
                  - List every retrieved game.
                  - Do not omit or add games.
                  - Preserve the retrieval ranking/order.

                  2. Common Community-Highlighted Gameplay Characteristics:
                  - Identify characteristics supported by at least two retrieved games.
                  - Use the Community Tag Frequency as supporting evidence.
                  - Report the frequency exactly as provided.
                  - Briefly identify which retrieved games support each characteristic.
                  - Do not repeat lengthy descriptions or explanations when the frequency and supporting games are sufficient.
                  - Group closely related tags when appropriate, but do not change their underlying frequency values.
                  - Do not infer characteristics from titles, franchises, reputation, or pretrained knowledge.

                  3. Steam Features That Occur Frequently:
                  - Report the calculated frequency values exactly as provided.
                  - Do not calculate, modify, reinterpret, or infer frequency values.
                  - Use the Calculated Steam Feature Frequency data as the authoritative source for feature frequency.
                  - Do not infer a Steam feature from a game's genre, franchise, description, or other characteristics.
                  - Distinguish between features that occur frequently and features that are simply present in one or more games.

                  4. Potential Gameplay Features Worth Considering:
                  - Clearly label these as recommendations.
                  - Recommendations may use information beyond the retrieved data.
                  - Prioritize recommendations based on relevance to the proposed concept.
                  - Recommendations are suggestions, not requirements.
                  - Do not attribute a recommendation to a specific game unless that game's retrieved data explicitly supports the attribution.
                  - Creative recommendations are allowed, but do not present them as facts about the retrieved games.

                  Important:
                  - The retrieved games represent the complete set selected for analysis.
                  - Consider all retrieved games for Sections 1-3.
                  - Do not focus only on the highest-ranked games.
                  - If information is unavailable in the retrieved context, state that it is not available.

                  Proposed game concept:
                  {input_text}

                  Retrieved Steam games:
                  {game_context}

                  Community Tag Frequency from Top 100 Similar Games:
                  {tag_context}

                  Calculated Steam Feature Frequency from Top 100 Similar Games:
                  {feature_context}


                  Output Format:
                  - Return the response as valid JSON.
                  - The JSON must contain exactly these four top-level keys, in this order:
                      1. "retrieved_similar_games"
                      2. "common_community_highlighted_gameplay_characteristics"
                      3. "frequent_steam_features"
                      4. "potential_gameplay_features_worth_considering"
                  - Do not include any additional top-level keys.
                  - Preserve the retrieval ranking/order in "retrieved_similar_games".
                  - Do not include Markdown code fences such as ```json.
                  """

    # context = game_context + "\n" + feature_context


    show_memory("After prompt created")

    USE_TEST_RESPONSE = True

    if USE_TEST_RESPONSE:
        response = test_gemini_response()
    else:
        response = generate_response_gemini(prompt)
        
            
    show_memory("After Gemini response")

    return (
        response,
        list_features,
        feature_group_results,
        tag_frequency,
        rag_results,
        market_data,
        popular_similar_games
    )


def dashboard_section(title, content):
    st.markdown(
        f"""
        <div class="dashboard-section">
            <h3>{title}</h3>
            <div class="dashboard-content">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def price_histogram(market_data):
    price_data = market_data[(market_data["initial_price"].notna())].copy()
    
    price_data["Price"] = np.where(
        price_data["is_free"] == 1,
        "Free",
        np.where(
            (price_data["coming_soon"] == 0) &
            (price_data["is_free"] == 0) &
            (price_data["initial_price"] == 0),
            "Not Available",
            price_data["initial_price"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "Unknown"
            )
        )
    )

    price_counts = (
        price_data["Price"]
        .value_counts()
        .reset_index()
    )

    price_counts.columns = ["Price", "Games"]

     # Separate Free, Not Available, and paid prices
    free_data = price_counts[price_counts["Price"] == "Free"]
    not_available_data = price_counts[price_counts["Price"] == "Not Available"]
    paid_data = price_counts[price_counts["Price"] != "Free"].copy()
    st.write(free_data)
    st.write(not_available_data)
    st.write(paid_data)

    paid_data["sort_price"] = (paid_data["Price"].str.replace("$","", regex=False).astype(float))

    paid_data = paid_data.sort_values("sort_price")
    price_counts = pd.concat([free_data,not_available_data,paid_data])



    figure = px.bar(price_counts, x="Price", y="Games",title="Price Distribution Among Similar Games", labels={"Price": " Price($)", "Games": "Number of Games"})
    # figure.update_layout(
    #        xaxis=dict(
    #                 categoryorder="array",
    #                 categoryarray=price_counts["Price"].tolist()
    #            )
    # )
    figure.update_traces(width=1.0)
    figure.update_xaxes(
        tickmode="array",
        tickvals=[4.99,9.99, 14.99, 19.99, 29.99, 39.99, 49.99, 59.99, 69.99],
        ticktext=["$4.99","$9.99", "$14.99","$19.99", "$29.99", "$39.99","49.99","59.99","69.99"]
    )
    st.plotly_chart(figure, use_container_width=True)



def market_summary(df):
    market_data = {}

    market_data['average_price'] = df['initial_price'].mean()
    market_data['median_price'] = df['initial_price'].median()
    market_data['minimum_price'] = df['initial_price'].min()
    market_data['maximum_price'] = df['initial_price'].max()

    # market_data = df[["appid",'initial_price','current_price','is_free','coming_soon','release_year','release_month','release_day']]
    return market_data
#----------------Streamlit----------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Game Development MVP Planning Tool", page_icon="placeholder",layout="wide")


st.title("Game Development MVP Planning Assistant")

st.write("""Enter a game concept to identify similar Steam games, common gameplay
            characteristics, frequently occurring Steam features, and potential features to consider for an MVP.
          """)

st.divider()

st.subheader("Game Concept")


with st.sidebar:

    #Filters to implement weight adjustment for semantic score and tag similarity
    #Possibly incorporate popularity (recommendations as an optional filter)
    with st.expander("Adjustments", expanded=False):
        similarity_weight = st.slider("Semantic Similarity", 0, 100, 70)
        top_games = st.slider("Number of Games", 3,20,5)

    tag_weight = 100 - similarity_weight

    st.write(f"Semantic Similarity: {similarity_weight}%")
    st.write(f"Community Tags: {tag_weight}%")
    st.write(f"Games Displayed: {top_games}")
    # selected_attribute = st.selectbox("Attribute: ", 1,index=0)

user_concept_input = st.text_area("Describe a game concept:", height=150, placeholder="Enter here...")

show_memory("Startup")

if st.button("Analyze Game Concept", type="primary"):
    if not user_concept_input.strip():
        st.warning("Please enter a game concept.")
    else:
        with st.spinner("Analyzing your game concept..."):
          response, list_features, feature_group_results, tag_frequency, rag_results, market_data, popular_similar_games = game_concept(
                  [user_concept_input],
                  similarity_weight,
                  tag_weight,
                  top_games
          )
        show_memory("End game_concept")
        if response is not None:
            games_response = response["retrieved_similar_games"]
            characteristics_response = response["common_community_highlighted_gameplay_characteristics"]
            features_response = response["frequent_steam_features"]
            recommendations_response = response["potential_gameplay_features_worth_considering"]
        else:
            recommendations_response = None



        display_results = rag_results[[
                              "appid",
                              "name",
                              "release_year",
                              "combined_score",
                              "tag_similarity",
                              "final_score"
        ]].copy()



        display_results = display_results.rename(columns={
            "name": "Games",
            "release_year": "Released",
            "combined_score": "Semantic Score",
            "tag_similarity": "Tag Similarity",
            "final_score": "Match Score"
        })

        display_results["Steam Link"] = ("https://store.steampowered.com/app/" + display_results["appid"].astype(int).astype(str)+ "/")

        group_chart_data = feature_group_results.copy()
        group_chart_data["Feature Group"] = (
            group_chart_data["feature_group"]
            .str.replace("_features", "", regex=False)
            .str.replace("_", " ")
            .str.title()
        )


        group_chart_data["Percentage"] = group_chart_data["Percentage"] * 100


        st.subheader("Top Similar Games:")
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.write("Titles Most Similar to Concept:")
            display_results["Match Score"] = (display_results["Match Score"].map(lambda x: f"{x:.0%}"))
            st.dataframe( display_results[["Games","Released", "Match Score", "Steam Link"]],
                    column_config={
                    "Steam Link": st.column_config.LinkColumn(
                        "Steam",
                        display_text="Open Steam"
                    )
                },
        
                width="stretch",
                hide_index=True
            )

            st.caption("Ranks games by their similarity to your proposed game concept based on the retrieval model.")

        with col2:
            # game_chart_data = rag_results.sort_values("final_score", ascending=False).copy()
            # game_chart_data = game_chart_data.rename(columns={"name": "Games", "final_score": "Match Score"})
            # game_chart_data["Match Score"] = game_chart_data["Match Score"] * 100
            # game_chart_data = game_chart_data[["Games", "Match Score"]]
            # st.bar_chart(game_chart_data,x="Games",y="Match Score", horizontal=True, sort="-Match Score")
            st.write("Most Engaged Similar Titles:")

            popular_similar_games = popular_similar_games.rename(columns={
                "name": "Game",
                "ccu": "Concurrent Active Users",
                "release_year": "Released",
                "review_count": "Steam Reviews"

            })


            popular_similar_games['Engagement Score'] = (popular_similar_games['Engagement Score'].map(lambda x: f"{x:.0%}"))
            popular_similar_games["Steam Link"] = ("https://store.steampowered.com/app/" + popular_similar_games["appid"].astype(int).astype(str)+ "/")

            st.dataframe(popular_similar_games[['Game','Released', 'Concurrent Active Users', 'Steam Reviews', 'Engagement Score', "Steam Link"]],
                         column_config={
                              "Steam Link": st.column_config.LinkColumn(
                              "Steam",
                              display_text="Open Steam"
                            )
                         },
                         width="stretch",
                         hide_index=True
                    )
            st.caption(
                "Ranks the most engaged games among the 100 most similar games retrieved. "
                "Engagement Score combines current concurrent players and Steam review volume to highlight games with stronger player engagement."
            )

        st.subheader("Market & Pricing Among Similar Games")
        price_histogram(market_data)

        price_data = market_summary(market_data)
        col1, col2, col3, col4  = st.columns(4)
        col1.metric(
            "Median Price",
            f"${price_data['median_price']:.2f}"
        )
        col2.metric(
            "Average Price",
            f"${price_data['average_price']:.2f}"
        )

        col3.metric(
            "Lowest Price",
            f"${price_data['minimum_price']:.2f}"
        )

        col4.metric(
            "Highest Price",
            f"${price_data['maximum_price']:.2f}"
        )


        st.subheader("Common Features Across Similar Games")

        st.caption("These features are commonly found among the games most similar to your concept.")

        col1, col2 = st.columns([1.2, 1])
        with col1:

            st.bar_chart(
                group_chart_data,
                x="Feature Group",
                y="Percentage",
                horizontal=True
            )
        with col2:
            st.write("Feature Breakdown")

            for group in feature_group_results["feature_group"]:

                group_data = list_features[
                    list_features["feature_group"] == group
                ].sort_values(
                    "percentage",
                    ascending=False
                )

                group_name = group.replace("_", " ").title()

                with st.expander(group_name):

                    display_data = group_data[
                        ["categories", "game_count", "percentage"]
                    ].copy()

                    display_data["percentage"] = (
                        display_data["percentage"]
                        .map(lambda x: f"{x:.0%}")
                    )

                    display_data = display_data.rename(
                        columns={
                            "categories": "Feature",
                            "game_count": "Similar Games",
                            "percentage": "Frequency"
                        }
                    )

                    st.dataframe(
                        display_data,
                        width="stretch",
                        hide_index=True
                    )





        st.subheader("Common Community Tags:")
        st.caption("Tags frequently associated with the games most similar to your concept.")

        tags_chart_data = tag_frequency.sort_values("game_frequency", ascending=False).head(15).copy()
        tags_chart_data = tags_chart_data.rename(columns={"tag_names": "Community Tags", "game_frequency": "Games"})
        st.bar_chart(tags_chart_data,x="Community Tags",y="Games", horizontal=True, sort="-Games")


        st.subheader("Potential Gameplay Features Worth Considering")
        # st.caption("Recommendations generated from the retrieved games and your proposed game concept.")
        st.caption(
            "AI-generated recommendations informed by the retrieved games "
            "and your proposed game concept. Recommendations are suggestions, "
            "not features directly observed in every retrieved game."
        )
        if recommendations_response is not None:
                for item in recommendations_response:
                      with st.container(border=True):
                          st.markdown(f"<p style='font-size:18px; font-weight:600; margin-bottom:4px;'> {item['recommendation']}</p>",unsafe_allow_html=True)
                          st.write(item["details"])
        else:
                st.warning("AI-generated recommendations are temporarily unavailable.")           

st.divider()
show_memory("End dashboard")
st.caption("MVP Tool for Game Development")
