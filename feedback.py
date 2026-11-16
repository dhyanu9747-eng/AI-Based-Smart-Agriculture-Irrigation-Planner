import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

FEEDBACK_FILE = Path("data/feedback.csv")


def feedback_page():
    st.title("💬 Feedback & Suggestions")
    st.write("Help us improve AgriSmart AI.")

    st.divider()

    with st.form("feedback_form"):
        st.subheader("Rate Your Experience")

        rating = st.select_slider(
            "How would you rate this application?",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x
        )

        category = st.selectbox(
            "Feedback Category",
            ["General Feedback", "Bug Report", "Feature Request", "Other"]
        )

        message = st.text_area(
            "Your Feedback",
            placeholder="Write your suggestions here..."
        )

        submitted = st.form_submit_button(
            "Submit Feedback",
            use_container_width=True
        )

        if submitted:
            if not message.strip():
                st.error("Please enter your feedback.")
            else:
                FEEDBACK_FILE.parent.mkdir(
                    parents=True, exist_ok=True
                )

                new_feedback = pd.DataFrame([{
                    "Username": st.session_state.get("username", "Unknown"),
                    "Rating": rating,
                    "Category": category,
                    "Message": message.strip(),
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])

                if FEEDBACK_FILE.exists():
                    old_feedback = pd.read_csv(FEEDBACK_FILE)
                    new_feedback = pd.concat(
                        [old_feedback, new_feedback],
                        ignore_index=True
                    )

                new_feedback.to_csv(FEEDBACK_FILE, index=False)

                st.success("Thank you! Your feedback has been submitted.")