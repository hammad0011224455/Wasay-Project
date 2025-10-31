"""Streamlit application entry point."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from .recommender_service import RecommenderService, UserProfile

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
COURSE_FILE = DATA_DIR / "courses.csv"
INTERACTION_FILE = DATA_DIR / "interactions.csv"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if COURSE_FILE.exists():
        courses = pd.read_csv(COURSE_FILE)
    else:
        courses = pd.DataFrame(
            [
                {"course_id": "C101", "title": "Intro to Python", "description": "Learn Python basics."},
                {"course_id": "C102", "title": "Advanced Data Science", "description": "Deep dive into ML."},
                {"course_id": "C103", "title": "Web Development", "description": "Build web apps."},
            ]
        )
    if INTERACTION_FILE.exists():
        interactions = pd.read_csv(INTERACTION_FILE)
    else:
        interactions = pd.DataFrame(
            [
                {"user_id": "U1", "course_id": "C101", "rating": 4.5},
                {"user_id": "U1", "course_id": "C102", "rating": 4.0},
                {"user_id": "U2", "course_id": "C103", "rating": 5.0},
            ]
        )
    return courses, interactions


def main() -> None:
    st.set_page_config(page_title="AI Course Recommender", layout="wide")
    st.title("AI-Based Online Course Recommender System")
    courses, interactions = load_data()
    service = RecommenderService(courses, interactions)
    service.train_models()

    users = sorted(interactions["user_id"].astype(str).unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", users)
    level = st.sidebar.selectbox("Skill Level", ["beginner", "intermediate", "advanced"], index=1)
    interest = st.sidebar.text_input("Primary Interest", "data science")
    recent_courses = st.sidebar.multiselect(
        "Courses you've enjoyed", courses["course_id"].astype(str).tolist(),
        default=[courses.iloc[0]["course_id"]],
    )

    if st.button("Get Recommendations"):
        profile = UserProfile(
            user_id=selected_user,
            level=level,
            interest=interest,
            recent_courses=recent_courses or [courses.iloc[0]["course_id"]],
        )
        recommendations = service.get_recommendations(profile)
        st.subheader("Top Recommendations")
        for rec in recommendations:
            with st.container():
                st.markdown(f"### {rec['title']} ({rec['course_id']})")
                st.write(rec["explanation"])
                feedback = st.radio(
                    f"Was this recommendation helpful for {rec['course_id']}?",
                    ("👍", "👎"),
                    key=f"feedback_{rec['course_id']}",
                    horizontal=True,
                )
                reward = 1.0 if feedback == "👍" else 0.0
                service.update_feedback(rec["course_id"], reward)
                st.write("Feedback recorded.")

    st.sidebar.markdown("---")
    st.sidebar.header("Debug Info")
    st.sidebar.write("Loaded courses:")
    st.sidebar.write(courses.head())
    st.sidebar.write("Loaded interactions:")
    st.sidebar.write(interactions.head())


if __name__ == "__main__":
    main()
