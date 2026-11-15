
import streamlit as st
import pickle
import pandas as pd
import requests
import html
from huggingface_hub import hf_hub_download

# PAGE CONFIGURATION

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# TMDB API CONFIGURATION

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "TMDB API key is missing. Add TMDB_API_KEY to "
        "Streamlit Community Cloud → App settings → Secrets."
    )
    st.stop()

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500/"

# CUSTOM CSS WITH ANIMATIONS

st.markdown("""
<style>

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes subtleGlow {
    0% {
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }
    50% {
        box-shadow: 0 12px 50px rgba(29, 78, 216, 0.15);
    }
    100% {
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }
}

.stApp {
    background: linear-gradient(135deg, #090e1b 0%, #111827 55%, #17152b 100%);
    color: #ffffff;
    animation: fadeInUp 0.8s ease-out;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    padding: 45px 35px;
    border-radius: 22px;
    background: linear-gradient(110deg, #130924, #0b1326, #070b12);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 30px;
    animation: fadeInUp 1s ease-out, subtleGlow 6s infinite ease-in-out;
}

.hero h1 {
    font-size: 43px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero .tagline,
.hero .tagline * {
    color: #FFD700 !important;
    -webkit-text-fill-color: #FFD700 !important;
}

.section-title {
    font-family: 'Poppins', sans-serif !important;
    font-size: 25px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 25px;
    margin-bottom: 15px;
    letter-spacing: 0.3px;
}

.movie-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 9px;
    min-height: 42px;
}

.movie-info {
    font-size: 12px;
    color: #aeb8cc;
    margin-bottom: 8px;
}

.movie-rating {
    color: #ffc857;
    font-size: 14px;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(90deg, #1d4ed8, #0284c7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 10px 20px rgba(29, 78, 216, 0.4);
    color: white;
    border: none;
}

div[data-baseweb="select"] > div {
    background-color: #20283a;
    border: 1px solid #444d63;
    border-radius: 10px;
    color: white;
    transition: all 0.3s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: #0284c7;
}

div[data-testid="stSelectbox"] label p {
    color: #ffffff !important;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s ease;
}

div[data-testid="stImage"]:hover img {
    transform: scale(1.04) translateY(-4px);
    box-shadow: 0 15px 30px rgba(2, 132, 199, 0.35);
}

.footer {
    text-align: center;
    color: #7e8799;
    padding: 30px 0 10px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# FETCH MOVIE DETAILS FROM TMDB

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None


# LOAD DATASET AND SIMILARITY MATRIX

HF_REPO_ID = "Sakib12341234/CineMatch-models"
HF_SIMILARITY_FILENAME = "similarity.pkl"


@st.cache_resource(
    show_spinner="Loading CineMatch model... This may take a few minutes on the first run."
)
def load_data():

    # Load movie dataset from GitHub
    with open("movie_list.pkl", "rb") as file:
        movies_dict = pickle.load(file)

    movie_data = pd.DataFrame(movies_dict)

    # Download similarity matrix from Hugging Face
    similarity_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_SIMILARITY_FILENAME,
        repo_type="model"
    )

    # Load downloaded similarity matrix
    with open(similarity_path, "rb") as file:
        similarity_data = pickle.load(file)

    # Validate dataset and similarity matrix
    if len(movie_data) != len(similarity_data):
        raise ValueError(
            "The movie dataset and similarity matrix do not match. "
            "Please verify that both files were generated from "
            "the same dataset and in the same order."
        )

    return movie_data, similarity_data


# INITIALIZE DATA

try:
    movies, similarity = load_data()

except Exception as error:
    st.error(
        "CineMatch could not load its movie data or similarity model."
    )
    st.exception(error)
    st.stop()


# RECOMMENDATION FUNCTION

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:

        movie_id = int(movies.iloc[i[0]].movie_id)

        movie_details = fetch_movie_details(movie_id)

        if movie_details:

            recommended_movies.append(movie_details)

        else:

            recommended_movies.append({
                "id": movie_id,
                "title": movies.iloc[i[0]].title,
                "poster_path": None,
                "release_date": "",
                "vote_average": 0,
                "overview": "",
                "genres": []
            })

    return recommended_movies


# HERO SECTION

st.markdown("""
<div class="hero">
<h1>🎬 CineMatch</h1>
<p style="color: #ffffff !important;">
Discover movies you'll love.
Explore personalized recommendations powered by machine learning.
</p>
<p class="tagline" style="
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
">
    ✦ DISCOVER &nbsp; ✦ EXPLORE &nbsp; ✦ ENJOY
</p>
</div>
""", unsafe_allow_html=True)


# MOVIE SELECTION

st.markdown(
    '<div class="section-title">🔎 Find Your Next Favourite Movie</div>',
    unsafe_allow_html=True
)

selected_movie_name = st.selectbox(
    "Choose a movie you already love",
    movies["title"].values,
    index=None,
    placeholder="Search or select a movie..."
)


# RECOMMENDATION BUTTON

if st.button("🎥 Discover Similar Movies", use_container_width=True):

    if selected_movie_name:

        with st.spinner("Finding movies you'll love... 🎬"):

            st.session_state.recommendations = recommend(
                selected_movie_name
            )

            st.session_state.selected_movie = selected_movie_name

    else:

        st.warning("Please select a movie first!")


# DISPLAY RECOMMENDATIONS

if "recommendations" in st.session_state:

    recommendations = st.session_state.recommendations

    st.markdown("---")

    st.markdown(
        f"""
        <div class="section-title">
            🍿 Because You Watched {html.escape(st.session_state.selected_movie)}
        </div>
        <p style="color:#aeb8cc;">
            Here are five movies selected based on content similarity.
        </p>
        """,
        unsafe_allow_html=True
    )

    if recommendations:

        columns = st.columns(5)

        for col, movie in zip(columns, recommendations):

            with col:

                poster_path = movie.get("poster_path")

                title = movie.get("title", "Unknown Movie")

                release_date = movie.get("release_date", "")

                rating = movie.get("vote_average", 0)

                overview = movie.get("overview", "")

                genres = movie.get("genres", [])

                # POSTER

                if poster_path:

                    poster_url = TMDB_IMAGE_URL + poster_path

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                            height:250px;
                            background:#252d40;
                            border-radius:12px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#aeb8cc;">
                            🎬 No Poster
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # MOVIE TITLE

                st.markdown(
                    f'<div class="movie-title">{html.escape(title)}</div>',
                    unsafe_allow_html=True
                )

                # RELEASE YEAR

                year = release_date[:4] if release_date else "N/A"

                st.markdown(
                    f'<div class="movie-info">📅 {year}</div>',
                    unsafe_allow_html=True
                )

                # RATING

                if rating:

                    st.markdown(
                        f'<div class="movie-rating">⭐ {rating:.1f}/10</div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.caption("Rating unavailable")

                # MOVIE DETAILS

                with st.expander("More details"):

                    if overview:
                        st.write(overview)
                    else:
                        st.caption("No overview available.")

                    if genres:

                        genre_names = [
                            genre["name"] for genre in genres
                        ]

                        st.write("**Genres:** " + ", ".join(genre_names))

    else:

        st.info("No recommendations found. Try another movie!")


# FOOTER

st.markdown("""
<div class="footer">
<h3 style="color:white;">🎬 CineMatch</h3>

<p>
Powered by Python • Streamlit • Machine Learning • TMDB API
</p>

<p>
This product uses the TMDB API but is not endorsed or certified by TMDB.
</p>
</div>
""", unsafe_allow_html=True)
