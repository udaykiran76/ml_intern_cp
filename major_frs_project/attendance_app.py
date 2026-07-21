import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import pickle
import os
import cv2
from datetime import datetime
from PIL import Image

DATABASE_FILE = "known_faces.pkl"
ATTENDANCE_FILE = "attendance.csv"

# ---------- Helper functions (same logic as before) ----------

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            return pickle.load(f)
    return {"names": [], "embeddings": []}

def save_database(database):
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)

def mark_attendance(name):
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    already_marked = ((df["Name"] == name) & (df["Date"] == today)).any()

    if already_marked:
        return f"{name} is already marked present today."
    else:
        new_entry = pd.DataFrame([{"Name": name, "Date": today, "Time": now.strftime("%H:%M:%S")}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return f"Attendance marked for {name} at {now.strftime('%H:%M:%S')}"

def recognize_face(image_rgb, database, tolerance=0.6):
    encodings = face_recognition.face_encodings(image_rgb)
    if len(encodings) == 0:
        return None, None
    new_embedding = encodings[0]
    known_embeddings = np.array(database["embeddings"])
    distances = face_recognition.face_distance(known_embeddings, new_embedding)
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]
    if best_distance <= tolerance:
        return database["names"][best_match_index], 1 - best_distance
    else:
        return "Unknown", 1 - best_distance

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Face Recognition Attendance", page_icon="🎓", layout="wide")
st.title("🎓 Face Recognition Attendance System")

tab1, tab2, tab3 = st.tabs(["📸 Take Attendance", "➕ Register New Person", "📊 Attendance Dashboard"])

# --- TAB 1: Take Attendance ---
with tab1:
    st.header("Mark Attendance")
    database = load_database()

    if len(database["names"]) == 0:
        st.warning("No one is registered yet. Go to the 'Register New Person' tab first.")
    else:
        camera_photo = st.camera_input("Look at the camera and take a photo")

        if camera_photo is not None:
            img = Image.open(camera_photo)
            img_rgb = np.array(img.convert("RGB"))

            name, confidence = recognize_face(img_rgb, database)

            if name is None:
                st.error("No face detected. Please try again.")
            elif name == "Unknown":
                st.error(f"Face not recognized (confidence too low: {confidence:.2f}). Attendance not marked.")
            else:
                st.success(f"Recognized: **{name}** (confidence: {confidence:.2f})")
                result = mark_attendance(name)
                st.info(result)

# --- TAB 2: Register New Person ---
with tab2:
    st.header("Register a New Person")
    new_name = st.text_input("Enter the person's name:")
    reg_photo = st.camera_input("Take a clear photo of their face")

    if st.button("Register"):
        if new_name.strip() == "":
            st.warning("Please enter a name first.")
        elif reg_photo is None:
            st.warning("Please take a photo first.")
        else:
            img = Image.open(reg_photo)
            img_rgb = np.array(img.convert("RGB"))
            encodings = face_recognition.face_encodings(img_rgb)

            if len(encodings) == 0:
                st.error("No face detected in the photo. Try again with better lighting.")
            else:
                database = load_database()
                database["names"].append(new_name)
                database["embeddings"].append(encodings[0])
                save_database(database)
                st.success(f"{new_name} registered successfully!")

# --- TAB 3: Dashboard ---
with tab3:
    st.header("Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)

        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Filter by date", value=None)
        with col2:
            selected_name = st.selectbox("Filter by name", options=["All"] + sorted(df["Name"].unique().tolist()))

        filtered_df = df.copy()
        if selected_date:
            filtered_df = filtered_df[filtered_df["Date"] == str(selected_date)]
        if selected_name != "All":
            filtered_df = filtered_df[filtered_df["Name"] == selected_name]

        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("Attendance Count per Person")
        st.bar_chart(df["Name"].value_counts())
    else:
        st.info("No attendance records yet.")
