import streamlit as st
import subprocess
import sys
import os
import time
import signal

# Page Config
st.set_page_config(page_title="Grok NBA Commentary", page_icon="🏀", layout="wide")

# Title and Header
st.title("🏀 Grok Real-Time NBA Commentary")
st.markdown("### AI-Powered Live Sports Narration")

# --- SESSION STATE MANAGEMENT ---
if "process" not in st.session_state:
    st.session_state.process = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# --- UI SECTION 1: CONFIGURATION ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        # 1. Upload YouTube Link
        youtube_url = st.text_input(
            "1. Paste YouTube Link", placeholder="https://www.youtube.com/watch?v=..."
        )

    with col2:
        # 2. Select Language
        language = st.selectbox(
            "2. Select Commentary Language", ["English", "Spanish", "French"], index=0
        )

# --- UI SECTION 2: DYNAMIC CONFIGURATION ---
if youtube_url:
    # --- LOGIC: Map URL to JSON File and Team Options ---

    # Default Fallback
    team_options = ["Home Team", "Away Team", "Neither (Neutral)"]
    input_json_file = "gameplay.json"

    # Case 1: Grizzlies vs Magic
    if "J8WABIinM64" in youtube_url:
        team_options = ["Memphis Grizzlies", "Orlando Magic", "Neither (Neutral)"]
        input_json_file = "magicvgrizzlies.json"

    # https://www.youtube.com/watch?v=It8h_JhEREw
    # Case 2: Lakers vs Clippers


    elif "L7o4UCIIqS4" in youtube_url:
        team_options = [
            "Milwuakee Bucks",
            "San Antonio Spurs",
            "Neither (Neutral)",
        ]
        input_json_file = "bucksvspurs.json"

    st.divider()
    st.subheader("3. Select Your Team")

    # 3. Team Support Selection
    team_support = st.radio(
        "Who do you want the commentators to support?",
        team_options,
        horizontal=True,
    )

    start_col, stop_col = st.columns([1, 5])

    with start_col:
        start_btn = st.button(
            "▶️ Start Broadcast", type="primary", use_container_width=True
        )

    with stop_col:
        if st.session_state.is_running:
            if st.button("⏹️ Stop", type="secondary"):
                if st.session_state.process:
                    st.session_state.process.terminate()
                    st.session_state.process = None
                st.session_state.is_running = False
                st.rerun()

    # --- EXECUTION LOGIC ---
    if start_btn:
        st.session_state.is_running = True

        # Layout: Video on Left, Live Logs on Right
        video_col, log_col = st.columns([3, 2])

        with video_col:
            st.info(f"Broadcast started! Supporting: **{team_support}**")

            # --- START TIME LOGIC ---
            start_time = 0
            # If Lakers vs Clippers, start at 2:58 (178 seconds)
            if "L7o4UCIIqS4" in youtube_url:
                start_time = 8

            # 2. Display YouTube (Muted + Autoplay + Start Time)
            st.video(youtube_url, autoplay=True, muted=True, start_time=start_time)

            # UX Warning for the edge case where autoplay fails
            st.caption(
                "⚠️ Note: If video does not autoplay, please click play immediately to sync with audio."
            )

        with log_col:
            st.write("🎙️ **Live Audio Feed**")
            log_placeholder = st.empty()

            # 3. Execute Script with Dynamic JSON File
            command = [
                sys.executable,
                "grok_script.py",
                "--input",
                input_json_file,
                "--pipeline",
                "--language",
                language,
                "--team_support",
                team_support,
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                st.session_state.process = process

                # Stream logs to the UI in real-time
                logs = []
                while True:
                    # Read line by line
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    if line:
                        clean_line = line.strip()
                        logs.append(clean_line)
                        # Keep only the last 15 lines for the log view
                        log_text = "\n".join(logs[-15:])
                        log_placeholder.code(log_text, language="bash")

            except Exception as e:
                st.error(f"Failed to start audio script: {e}")

else:
    st.info("Please enter a YouTube link to continue.")
