
import streamlit as st
import json
import hashlib
from pathlib import Path

USERS_FILE = Path("users.json")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_users(users):
    USERS_FILE.write_text(
        json.dumps(users, indent=4),
        encoding="utf-8"
    )


def authentication_page():

    st.markdown("""
    
    <style>
.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, #405b3d 0%, transparent 42%),
        radial-gradient(ellipse at 90% 90%, #594b32 0%, transparent 40%),
        linear-gradient(135deg, #17251c, #26392a 55%, #17251c);
    color: #f4f0df;
}

.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

.brand {
    text-align: center;
    padding: 18px 10px 10px;
}

.brand-icon {
    font-size: 58px;
}

.brand-title {
    color: #d5e8b5;
    font-size: 46px;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 5px 0;
}

.brand-subtitle {
    color: #c6c9b4;
    font-size: 13px;
    letter-spacing: 2px;
}

.hero-title {
    text-align: center;
    color: #f5f0df;
    font-size: 27px;
    font-weight: 700;
    margin-top: 25px;
}

.hero-description {
    text-align: center;
    color: #c6c9b4;
    font-size: 14px;
    margin-bottom: 25px;
}

div[data-testid="stTabs"] {
    background: rgba(239, 236, 216, 0.09);
    border: 1px solid rgba(221, 225, 196, 0.25);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(15px);
}

div[data-testid="stTabs"] button {
    color: #e8e5d5;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #d5e8a5;
}

div[data-testid="stTextInput"] label {
    color: #f0eddf;
    font-weight: 600;
}

div[data-testid="stTextInput"] input {
    background: #f0eee2;
    color: #263629;
    border: 1px solid #c5c9ad;
    border-radius: 10px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #b6c98a;
    box-shadow: 0 0 0 1px #b6c98a;
}

div.stButton > button {
    background: linear-gradient(100deg, #b6c98a, #d5dca5);
    color: #263629;
    border: none;
    border-radius: 10px;
    height: 46px;
    font-weight: 800;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #e1e8bd;
    color: #263629;
    transform: translateY(-2px);
}

.feature-row {
    text-align: center;
    color: #d5e8b5;
    font-size: 13px;
    margin-top: 25px;
    padding: 15px;
    border-top: 1px solid rgba(221, 225, 196, 0.25);
}

.footer {
    text-align: center;
    color: #aeb9a2;
    font-size: 12px;
    margin-top: 22px;
}
</style>
    
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🌱</div>
        <h1 class="brand-title">AgriSmart AI</h1>
        <div class="brand-subtitle">
            INTELLIGENT AGRICULTURE • SMART IRRIGATION • BETTER FUTURE
        </div>
    </div>

    <div class="hero-title">
        Grow Smarter. Save Water. Farm Better.
    </div>

    <div class="hero-description">
        Your intelligent partner for modern and sustainable agriculture.
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])

    with center:
        tab1, tab2 = st.tabs(["🔐 LOGIN", "📝 REGISTER"])

        with tab1:
            st.markdown("### Welcome Back!")
            st.caption("Sign in to access your smart farming dashboard.")

            username = st.text_input(
                "Username",
                key="login_username",
                placeholder="Enter your username"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="Enter your password"
            )

            if st.button("🚀 Login to Dashboard", use_container_width=True):
                users = load_users()

                if (
                    username in users
                    and users[username]["password"] == hash_password(password)
                ):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab2:
            st.markdown("### Create Your Account")
            st.caption("Join AgriSmart AI and manage your farm intelligently.")

            new_username = st.text_input(
                "Choose Username",
                key="register_user",
                placeholder="Create a username"
            )

            new_password = st.text_input(
                "Create Password",
                type="password",
                key="register_password",
                placeholder="Minimum 8 characters"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="confirm_password",
                placeholder="Enter password again"
            )

            if st.button("🌱 Create My Account", use_container_width=True):
                users = load_users()

                if not new_username.strip() or not new_password:
                    st.error("Please enter a username and password.")
                elif new_username in users:
                    st.error("Username already exists.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 8:
                    st.error("Password must contain at least 8 characters.")
                else:
                    users[new_username] = {
                        "password": hash_password(new_password),
                        "role": "user"
                    }
                    save_users(users)
                    st.success("Account created! You can now log in.")

    st.markdown("""
    <div class="feature-row">
        🌦️ Live Weather &nbsp; • &nbsp;
        🤖 AI Predictions &nbsp; • &nbsp;
        💧 Smart Irrigation &nbsp; • &nbsp;
        📊 Farm Analytics
    </div>

    <div class="footer">
        AgriSmart AI | Smart Agriculture Management System
        <br>Developed as an AI & ML academic project
    </div>
    """, unsafe_allow_html=True)
