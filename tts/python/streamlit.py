import streamlit as st
import subprocess
import os
import signal
import atexit
import asyncio
import json
import time

# Import streaming functions from grok_script
from grok_script import (
    NBACommentaryAgent,
    merge_close_events,
    sanitize_text,
    stream_to_speaker,
)

# Page Config
st.set_page_config(page_title="Grok NBA Commentary", page_icon="🏀", layout="wide")

# Title and Header
st.title("🏀 Grok Real-Time NBA Commentary")
st.markdown("### AI-Powered Live Sports Narration")

# --- CLEANUP LOGIC ---
PID_FILE = "crowd_pid.txt"


def kill_all_processes():
    """
    Forcefully kills ALL background processes on refresh/exit.
    1. Kills 'afplay' (Crowd Noise + TTS Audio)
    2. Kills 'grok_script.py' (The Commentary Engine)
    """
    # 1. Kill Audio Players
    try:
        subprocess.run(["pkill", "-f", "afplay"], check=False)
    except Exception:
        pass

    # 2. Kill Commentary Script
    try:
        # -f matches the command line name
        subprocess.run(["pkill", "-f", "grok_script.py"], check=False)
    except Exception:
        pass

    # 3. Cleanup PID file if it exists
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.killpg(pid, signal.SIGTERM)
            os.remove(PID_FILE)
        except Exception:
            pass


# 1. Register cleanup on server exit (Ctrl+C)
atexit.register(kill_all_processes)

# 2. Run cleanup on startup (Handle page refreshes)
# This effectively "Restarts" the backend state when the frontend reloads.
if "cleanup_done" not in st.session_state:
    kill_all_processes()
    st.session_state.cleanup_done = True


# --- SESSION STATE MANAGEMENT ---
if "process" not in st.session_state:
    st.session_state.process = None
if "crowd_process" not in st.session_state:
    st.session_state.crowd_process = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "stop_streaming" not in st.session_state:
    st.session_state.stop_streaming = False


# --- STREAMING COMMENTARY FUNCTION ---
async def run_streaming_commentary(events, language, team_support, log_placeholder):
    """
    Run streaming pipeline with real-time Streamlit UI updates.

    Flow: NBA Event -> Grok LLM (token stream) -> TTS (audio chunks) -> Speaker
    """
    agent = NBACommentaryAgent(language=language, team_support=team_support)
    optimized = merge_close_events(events, threshold=3)  # Combine events within 5s

    voices = ["leo", "ara", "rex"]
    voice_idx = 0
    logs = []
    start_time = time.time()  # Track when broadcast started

    for i, event in enumerate(optimized):
        # Check for cancellation
        if st.session_state.stop_streaming:
            logs.append("⏹️ Streaming stopped by user")
            log_placeholder.code("\n".join(logs[-15:]), language="bash")
            break

        event_time = event.get("timeActual", 0)
        description = event.get("description", "")[:50]

        # Update UI with current event
        logs.append(f"[{event_time:.1f}s] {description}...")
        log_placeholder.code("\n".join(logs[-15:]), language="bash")

        # 1. Stream tokens from Grok LLM
        text = ""
        try:
            async for token in agent.process_event_streaming(event):
                text += token
                # Update UI with streaming tokens
                current_log = logs[-1] if logs else ""
                display_logs = logs[:-1] + [f"{current_log}\n  LLM: {text}"]
                log_placeholder.code("\n".join(display_logs[-15:]), language="bash")
        except Exception as e:
            logs.append(f"  LLM Error: {e}")
            log_placeholder.code("\n".join(logs[-15:]), language="bash")
            continue

        # 2. Sanitize text
        clean_text = sanitize_text(text)
        if not clean_text:
            logs.append("  (empty, skipping)")
            continue

        # 3. WAIT until event_time before playing TTS (sync with video)
        elapsed = time.time() - start_time
        wait_needed = event_time - elapsed
        if wait_needed > 0:
            logs.append(f"  ⏳ Waiting {wait_needed:.1f}s until [{event_time:.1f}s]...")
            log_placeholder.code("\n".join(logs[-15:]), language="bash")
            await asyncio.sleep(wait_needed)

        # 4. Stream to speaker via WebSocket TTS (at correct time)
        current_voice = voices[voice_idx % len(voices)]
        voice_idx += 1

        logs.append(f"  🔊 [{current_voice}]: {clean_text[:40]}...")
        log_placeholder.code("\n".join(logs[-15:]), language="bash")

        try:
            await stream_to_speaker(clean_text, voice=current_voice)
        except Exception as e:
            logs.append(f"  TTS Error: {e}")
            log_placeholder.code("\n".join(logs[-15:]), language="bash")

    logs.append("✅ Streaming complete")
    log_placeholder.code("\n".join(logs[-15:]), language="bash")


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

    # Case 2: Bucks vs Spurs
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
                # 1. Set cancellation flag for streaming
                st.session_state.stop_streaming = True

                # 2. Kill Crowd Noise (Kill Process Group to ensure 'while' loop ends)
                if st.session_state.crowd_process:
                    try:
                        os.killpg(
                            os.getpgid(st.session_state.crowd_process.pid),
                            signal.SIGTERM,
                        )
                    except ProcessLookupError:
                        pass
                    st.session_state.crowd_process = None

                # 3. Kill any remaining audio players
                try:
                    subprocess.run(["pkill", "-f", "afplay"], check=False)
                except Exception:
                    pass

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
            if "L7o4UCIIqS4" in youtube_url:
                start_time = 8

            # 2. Display YouTube (Muted + Autoplay + Start Time)
            st.video(youtube_url, autoplay=True, muted=True, start_time=start_time)


        with log_col:
            st.write("🎙️ **Live Audio Feed**")
            log_placeholder = st.empty()

            # --- PROCESS 1: BACKGROUND CROWD NOISE ---
            # We use a shell loop to play the MP3 indefinitely.
            # -v 0.1 sets the volume to 10% so it doesn't overpower the commentary.
            try:
                crowd_cmd = "while true; do afplay -v 0.1 basketballcrowd.mp3; done"
                crowd_process = subprocess.Popen(
                    crowd_cmd,
                    shell=True,
                    preexec_fn=os.setsid,  # Create new process group for easier cleanup
                )
                st.session_state.crowd_process = crowd_process
            except Exception as e:
                st.error(f"Failed to start crowd noise: {e}")

            # --- PROCESS 2: STREAMING COMMENTARY (In-Process) ---
            # Reset stop flag
            st.session_state.stop_streaming = False

            try:
                # Load events from JSON
                with open(input_json_file) as f:
                    events = json.load(f)

                # Run streaming pipeline in-process
                asyncio.run(
                    run_streaming_commentary(
                        events, language, team_support, log_placeholder
                    )
                )

            except Exception as e:
                st.error(f"Failed to start streaming commentary: {e}")

else:
    st.info("Please enter a YouTube link to continue.")
