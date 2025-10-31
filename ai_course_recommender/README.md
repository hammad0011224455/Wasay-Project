# AI-Based Online Course Recommender System

A hybrid recommender system that combines content-based filtering, collaborative filtering, and reinforcement learning to provide personalized online course suggestions.

## Features
- ✅ Context-aware recommendations
- ✅ Explainable AI
- ✅ Reinforcement learning adaptation
- ✅ Streamlit interactive app

## Project Structure
```
ai_course_recommender/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── utils/
├── app/
└── models/saved/
```

## Datasets
- Open University Learning Analytics Dataset (OULAD)
- Kaggle MOOCs Recommendation Dataset
- Simulated datasets for rapid experimentation

Place processed CSV files under `data/processed/` named `courses.csv` and `interactions.csv`.

## Model Overview
- **Content-Based**: TF-IDF and BERT embeddings for course similarity.
- **Collaborative Filtering**: Matrix Factorization (SVD) and Neural Collaborative Filtering (PyTorch).
- **Hybrid**: Weighted combination of content and collaborative scores.
- **Reinforcement Learning**: Contextual bandit for adaptive personalization.

## Evaluation
Metrics implemented in `utils/evaluation.py`:
- Precision@K
- Recall@K
- NDCG
- Mean Reciprocal Rank (MRR)

The module also includes an A/B testing simulator to compare model variants.

## Explainability
`utils/explainability.py` generates user-friendly rationales, e.g.
> "Recommended because you liked 'Python for Data Science' and have shown interest in similar intermediate-level courses."

## Streamlit App
Run the interactive demo:
```bash
pip install -r requirements.txt
streamlit run app/main.py
```
The app provides user selection, recommendations, explanations, and feedback capture for reinforcement learning updates.

## GitHub Actions
Continuous integration workflow in `.github/workflows/ci.yml` performs linting and unit testing.

## License
Released under the MIT License. See [LICENSE](LICENSE).
