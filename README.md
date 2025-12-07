# 🏀 GrokCast NBA

**AI-Powered Real-Time NBA Commentary — Personalized, Multilingual, and Passionate**

Turn any NBA game into your personal broadcast with an AI commentator that roots for YOUR team, speaks YOUR language, and never misses a beat.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![xAI](https://img.shields.io/badge/Powered%20by-xAI-purple.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

---

## ✨ Features

- **🎙️ Real-Time Commentary** — Play-by-play audio synced precisely with game events
- **❤️ Homer Mode** — Pick your team and get biased, passionate commentary
- **🌍 Multilingual** — English, Spanish, and French with native fluency
- **⚡ True Streaming** — Audio starts playing while AI is still thinking
- **🔍 Live Player Stats** — X Search fills quiet moments with real statistics
- **🔄 Smart Interrupts** — Breaking plays override current commentary instantly

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  NBA Play Data  │ ──▶ │   Grok 4.1 LLM  │ ──▶ │  Grok Voice TTS │ ──▶ │   Speaker   │
│     (JSON)      │     │    (tokens)     │     │   (WebSocket)   │     │   (audio)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │    X Search     │
                        │  (filler stats) │
                        └─────────────────┘
```

### How It Works

1. **Event Arrives** → NBA play-by-play data triggers at the correct timestamp
2. **Grok Generates** → Grok 4.1 creates excited, biased commentary (token by token)
3. **Voice Speaks** → Grok Voice converts to natural speech via WebSocket streaming
4. **Interrupt Ready** → New events can override current speech instantly
5. **Filler Mode** → During quiet moments, X Search finds player stats to share

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Grok 4.1 (fast non-reasoning) | Generate context-aware commentary |
| **Search** | Grok X Search | Fetch real-time player statistics |
| **Voice** | Grok Voice (WebSocket TTS) | Natural multilingual speech |
| **Framework** | LangChain + OpenAI SDK | LLM orchestration |
| **UI** | Streamlit | Web interface |
| **Audio** | PyAudio | Real-time audio playback |
| **Data** | nba_api | Official NBA play-by-play |

---

## 📁 Project Structure

```
tts/python/
├── streamlit.py          # 🎮 Main web interface
├── grok_script.py        # 🧠 Core AI engine
│   ├── NBACommentaryAgent    # Commentary generation
│   ├── search_player_stats() # X Search integration
│   └── stream_tokens_to_speaker() # Real-time TTS
├── requirements.txt      # 📦 Dependencies
├── magicvgrizzlies.json  # 🏀 Sample game data
└── ARCHITECTURE.md       # 📐 System diagrams
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- xAI API Key ([Get one here](https://x.ai))
- PortAudio (for PyAudio)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/grokcast-nba.git
cd grokcast-nba/tts/python

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install streamlit langchain-xai openai pydantic

# macOS: Install PortAudio for audio playback
brew install portaudio
```

### Configuration

Create a `.env` file in the `tts/python` directory:

```env
XAI_API_KEY=your_xai_api_key_here
GROK_MODEL=grok-4-1-fast-non-reasoning
```

### Run the App

```bash
cd tts/python
streamlit run streamlit.py
```

Open http://localhost:8501 in your browser.

---

## 🎮 Usage

1. **Upload Game Data** — Select a JSON file with NBA play-by-play events
2. **Pick Your Team** — Choose which team to root for (or stay neutral)
3. **Select Language** — English, Spanish, or French
4. **Start Broadcast** — Click play and enjoy personalized commentary!

### Sample Output

```
🚀 Real-time simulation started (608 events)
   Teams: MEM, ORL | Players: 24

[13.8s] 🎤 [leo] D. Bane driving floating Jump Shot
  💬 Bane floats it in! Beautiful!
  ✓ Done

📊 [Filler] [eve] Looking up J. Jackson Jr....
  💬 Jackson averaging 22 points this month!
  📊 Filler interrupted by event

[37.3s] 🎤 [leo] MISS K. Caldwell-Pope 3PT
  💬 Caldwell-Pope... no good!
  ✓ Done
```

---

## 🔧 Key Components

### NBACommentaryAgent

Generates biased, exciting commentary based on team preference:

```python
agent = NBACommentaryAgent(
    language="en",           # en, sp, fr
    team_support="Grizzlies" # Your team (or "Neither")
)

# Stream tokens for real-time TTS
async for token in agent.process_event_streaming(event):
    print(token, end="")
```

### Token-by-Token Streaming

Audio starts playing **while the LLM is still generating**:

```python
await stream_tokens_to_speaker(
    agent.process_event_streaming(event),
    voice="leo"  # leo, eve, ara, rex, sal, una
)
```

### X Search Filler

During quiet moments, searches for real player stats:

```python
async for token in search_player_stats("LeBron James", ["LAL", "BOS"]):
    print(token, end="")
# Output: "LeBron averaging 25 points in his last 5 games!"
```

---

## 🎯 API Reference

### Voices Available

| Voice | Style |
|-------|-------|
| `leo` | Energetic male |
| `eve` | Enthusiastic female |
| `ara` | Professional |
| `rex` | Deep, dramatic |
| `sal` | Casual |
| `una` | Warm |

### Languages Supported

- `en` — English
- `sp` — Spanish (Español)
- `fr` — French (Français)

---

## 🏆 What Makes This Special

1. **True Real-Time** — Not batch processing; tokens stream directly to voice
2. **Smart Interrupts** — Breaking plays override current speech seamlessly
3. **Homer Bias** — AI genuinely roots for your team with emotional reactions
4. **X Search Integration** — Real statistics during downtime, not made-up filler
5. **100% xAI Powered** — Grok LLM + Grok Voice + X Search end-to-end

---

## 📜 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **xAI** — For Grok 4.1, Grok Voice, and X Search APIs
- **NBA** — For the incredible game data
- **Streamlit** — For the simple, powerful UI framework

---

**Built with ❤️ for basketball fans everywhere**

*GrokCast NBA — Your team. Your language. Your broadcast.* 🏀🎙️
