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
# We use session state to keep track of the background process
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

# --- UI SECTION 2: TEAM SELECTION (Conditional) ---
if youtube_url:
    st.divider()
    st.subheader("3. Select Your Team")

    # 3. Team Support Selection
    team_support = st.radio(
        "Who do you want the commentators to support?",
        ["Memphis Grizzlies", "Orlando Magic", "Neither (Neutral)"],
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

            # 2. Display YouTube (Muted + Autoplay)
            # Note: Browser policies may block autoplay if not muted.
            # We explicitly set muted=True to ensure autoplay works.
            st.video(youtube_url, autoplay=True, muted=True)

        with log_col:
            st.write("🎙️ **Live Audio Feed**")
            log_placeholder = st.empty()

            # 3. Execute Script
            # We add the --language argument dynamically based on user selection
            command = [
                sys.executable,
                "grok_script.py",
                "--input",
                "gameplay.json",
                "--pipeline",
                "--language",
                language,  # Pass the variable from st.selectbox
            ]

            # Pass environment variables if needed (e.g. for language/team)
            # env = os.environ.copy()
            # env["USER_TEAM"] = team_support

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
