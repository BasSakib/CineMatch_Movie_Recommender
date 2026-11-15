
# 🎬 CineMatch

A movie recommendation web app built with **Python** and **Streamlit**. Uses content-based filtering to recommend five similar movies based on your selection.

## 🚀 Live Demo
[Try CineMatch](https://cinematchmovierecommender-ak4tytqpwenlpkpqzjk2eo.streamlit.app)

## ✨ Features
- Personalized movie recommendations
- Movie posters, ratings, and details
- Interactive dark-themed interface
- TMDB API integration

## 🛠️ Technologies
Python · Streamlit · Pandas · Scikit-learn · TMDB API

## 📊 Dataset
[TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 💻 Run Locally

```bash
git clone https://github.com/BasSakib/CineMatch_Movie_Recommender.git
cd CineMatch_Movie_Recommender
pip install -r requirements.txt
streamlit run Main.py
```

Configure your TMDB API key in `.streamlit/secrets.toml`:

TMDB_API_KEY = "YOUR_TMDB_API_KEY"


The large `similarity.pkl` file is hosted separately on [Hugging Face](https://huggingface.co/Sakib12341234/CineMatch-models) and downloaded automatically.

## 🔗 Links
- [Live Demo](https://cinematchmovierecommender-ak4tytqpwenlpkpqzjk2eo.streamlit.app)
- [GitHub Repository](https://github.com/BasSakib/CineMatch_Movie_Recommender)
- [Kaggle Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Hugging Face Model](https://huggingface.co/Sakib12341234/CineMatch-models)

This product uses the TMDB API but is not endorsed or certified by TMDB.
