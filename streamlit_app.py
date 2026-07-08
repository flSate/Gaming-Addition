import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Gaming Addiction Risk Predictor",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Gaming Addiction Risk Predictor")
st.markdown("Predict a user's **Mental Health Risk Score** based on gaming habits and lifestyle.")

# -------------------------------
# LOAD MODEL COMPONENTS
# -------------------------------

@st.cache_resource
def load_components():
    return joblib.load("full_model_pipeline_components.joblib")

components = load_components()

encoder = components["encoder"]
scaler = components["scaler"]
selector = components["feature_selector"]
selected_feature_names = components["selected_feature_names"]
linear_model = components["model"]
columns_to_drop_from_features = components["columns_to_drop_from_features"]
numerical_cols_df = components["numerical_cols_df"]
training_columns = components["training_columns"]

# -------------------------------
# INPUT LAYOUT
# -------------------------------

col1, col2, col3 = st.columns(3)

# ==========================================
# COLUMN 1
# ==========================================

with col1:

    st.subheader("👤 Personal Information")

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=24
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female","Non-binary"]
    )

    country = st.text_input(
        "Country",
        value="USA"
    )

    occupation = st.text_input(
        "Occupation",
        value="Student"
    )

    income_level = st.selectbox(
        "Income Level",
        [
            "Low",
            "Lower-Middle",
            "Middle",
            "Upper-Middle",
            "High"
        ]
    )

    relationship_status = st.selectbox(
        "Relationship Status",
        [
            "Single",
            "In a relationship",
            "Married"
        ]
    )

# ==========================================
# COLUMN 2
# ==========================================

with col2:

    st.subheader("🎮 Gaming Information")

    years_gaming = st.number_input(
        "Years Gaming",
        value=7
    )

    preferred_genre = st.text_input(
        "Preferred Genre",
        value="RPG"
    )

    platform = st.selectbox(
        "Platform",
        [
            "PC",
            "Console",
            "Mobile"
        ]
    )

    device_type = st.text_input(
        "Device Type",
        value="Desktop"
    )

    rank_tier = st.text_input(
        "Rank Tier",
        value="Silver"
    )

    subscription_status = st.selectbox(
        "Subscription",
        [
            "Active",
            "Inactive",
            "None"
        ]
    )

    behavioral_cluster = st.text_input(
        "Behavioral Cluster",
        value="Casual Enjoyer"
    )

# ==========================================
# COLUMN 3
# ==========================================

with col3:

    st.subheader("🧠 Lifestyle")

    daily_playtime_hours = st.slider(
        "Daily Playtime (Hours)",
        0.0,24.0,6.0
    )

    weekly_play_sessions = st.number_input(
        "Weekly Play Sessions",
        value=8
    )

    late_night_sessions_hours = st.slider(
        "Late Night Sessions",
        0.0,12.0,2.5
    )

    weekend_playtime_hours = st.slider(
        "Weekend Playtime",
        0.0,48.0,7.0
    )

    consecutive_hours_max = st.slider(
        "Longest Session",
        0.0,24.0,9.0
    )

st.divider()

st.subheader("📊 Additional Metrics")

c1,c2,c3,c4 = st.columns(4)

with c1:

    multiplayer_ratio = st.slider(
        "Multiplayer Ratio",
        0.0,1.0,0.5
    )

    toxic_chat_reports = st.number_input(
        "Toxic Chat Reports",
        value=1
    )

    rage_quit_frequency = st.number_input(
        "Rage Quit Frequency",
        value=0
    )

    in_game_purchases = st.number_input(
        "In-game Purchases",
        value=2
    )

with c2:

    monthly_spending_usd = st.number_input(
        "Monthly Spending",
        value=30.0
    )

    lootbox_openings = st.number_input(
        "Lootboxes Opened",
        value=5
    )

    dopamine_dependency_index = st.slider(
        "Dopamine Dependency",
        0.0,1.0,0.6
    )

    self_control_score = st.slider(
        "Self Control",
        0.0,10.0,7.0
    )

with c3:

    impulsiveness_score = st.slider(
        "Impulsiveness",
        0.0,10.0,4.0
    )

    emotional_stability = st.slider(
        "Emotional Stability",
        0.0,10.0,6.0
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        0.0,12.0,7.0
    )

    exercise_frequency_per_week = st.slider(
        "Exercise / Week",
        0,7,3
    )

with c4:

    caffeine_intake_cups_day = st.number_input(
        "Caffeine Cups / Day",
        value=2
    )

    social_interaction_hours = st.slider(
        "Social Interaction",
        0.0,24.0,4.0
    )

    gpa_or_performance_score = st.number_input(
        "Performance Score",
        value=3.2
    )

    missed_deadlines = st.number_input(
        "Missed Deadlines",
        value=1
    )

    productivity_drop_percent = st.slider(
        "Productivity Drop %",
        0.0,100.0,10.0
    )

    absenteeism_days = st.number_input(
        "Absenteeism Days",
        value=2
    )

    internet_speed_mbps = st.number_input(
        "Internet Speed",
        value=150.0
    )

    screen_time_total_hours = st.slider(
        "Total Screen Time",
        0.0,24.0,8.0
    )

    burnout_probability = st.slider(
        "Burnout Probability",
        0.0,1.0,0.8
    )

    anxiety_level = st.slider(
        "Anxiety Level",
        0.0,10.0,5.0
    )

    depression_indicator = st.slider(
        "Depression Indicator",
        0.0,1.0,0.5
    )

    stress_score = st.slider(
        "Stress Score",
        0.0,10.0,6.0
    )

def predict_new_user(user_input_data):

    # Convert dictionary to DataFrame
    new_user_df = pd.DataFrame([user_input_data])

    # ----------------------------------
    # STEP 1 - Drop unused columns
    # ----------------------------------
    existing_cols_to_drop = [
        col for col in columns_to_drop_from_features
        if col in new_user_df.columns
    ]

    X_raw_new = new_user_df.drop(columns=existing_cols_to_drop)

    # ----------------------------------
    # STEP 2 - One-Hot Encode
    # ----------------------------------
    categorical_cols = X_raw_new.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    encoded = encoder.transform(X_raw_new[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_raw_new.index
    )

    X_processed = X_raw_new.drop(columns=categorical_cols)

    X_processed = pd.concat(
        [X_processed, encoded_df],
        axis=1
    )

    # ----------------------------------
    # STEP 3 - Match training columns
    # ----------------------------------

    for col in training_columns:
        if col not in X_processed.columns:
            X_processed[col] = 0

    X_processed = X_processed[training_columns]

    # ----------------------------------
    # STEP 4 - Scale numerical columns
    # ----------------------------------

    numerical_cols = [
        col for col in numerical_cols_df
        if col in X_processed.columns
    ]

    X_processed[numerical_cols] = scaler.transform(
        X_processed[numerical_cols]
    )

    # ----------------------------------
    # STEP 5 - Feature Selection
    # ----------------------------------

    X_selected = X_processed[selected_feature_names]

    # ----------------------------------
    # STEP 6 - Prediction
    # ----------------------------------

    prediction = linear_model.predict(X_selected)

    return prediction[0]

# -----------------------------------------
# BUILD USER INPUT
# -----------------------------------------

new_user_data = {

    "age": age,
    "gender": gender,
    "country": country,
    "occupation": occupation,
    "income_level": income_level,
    "years_gaming": years_gaming,
    "preferred_genre": preferred_genre,
    "platform": platform,
    "device_type": device_type,
    "rank_tier": rank_tier,

    "daily_playtime_hours": daily_playtime_hours,
    "weekly_play_sessions": weekly_play_sessions,
    "late_night_sessions_hours": late_night_sessions_hours,
    "weekend_playtime_hours": weekend_playtime_hours,
    "consecutive_hours_max": consecutive_hours_max,

    "multiplayer_ratio": multiplayer_ratio,
    "toxic_chat_reports": toxic_chat_reports,
    "rage_quit_frequency": rage_quit_frequency,

    "in_game_purchases": in_game_purchases,
    "monthly_spending_usd": monthly_spending_usd,
    "lootbox_openings": lootbox_openings,

    "subscription_status": subscription_status,

    "dopamine_dependency_index": dopamine_dependency_index,
    "self_control_score": self_control_score,
    "impulsiveness_score": impulsiveness_score,
    "emotional_stability": emotional_stability,

    "sleep_hours": sleep_hours,
    "exercise_frequency_per_week": exercise_frequency_per_week,
    "caffeine_intake_cups_day": caffeine_intake_cups_day,
    "social_interaction_hours": social_interaction_hours,

    "relationship_status": relationship_status,

    "gpa_or_performance_score": gpa_or_performance_score,
    "missed_deadlines": missed_deadlines,
    "productivity_drop_percent": productivity_drop_percent,
    "absenteeism_days": absenteeism_days,
    "internet_speed_mbps": internet_speed_mbps,
    "screen_time_total_hours": screen_time_total_hours,

    "behavioral_cluster": behavioral_cluster,

    "burnout_probability": burnout_probability,

    # Included because your original prediction
    # function expects them before dropping them.
    "anxiety_level": anxiety_level,
    "depression_indicator": depression_indicator,
    "stress_score": stress_score,
}

st.divider()

if st.button("🎮 Predict Mental Health Risk"):

    prediction = predict_new_user(new_user_data)

    st.success(
        f"Predicted Mental Health Risk Score: {prediction:.2f}"
    )