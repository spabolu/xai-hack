# NBA Commentary Agent

An AI-powered sports commentary agent that generates natural commentary from NBA play-by-play events using Grok 4 (xAI) and Langchain.

## Features

- 🤖 **Grok 4 Integration**: Uses xAI's Grok API for high-quality commentary generation
- 🧵 **Thread Management**: Maintains conversation context throughout the game
- 📊 **Structured Output**: Returns Pydantic-validated commentary for 1-2 commentators
- 🎙️ **TTS Ready**: Output formatted for text-to-speech voice models
- 🚀 **Simple & Clean**: Focused on developer experience with minimal complexity

## Setup

### Prerequisites

- Python 3.12+
- xAI API key ([Get one here](https://x.ai/))

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd xai-hack
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```bash
# xAI API Configuration
XAI_API_KEY=your_xai_api_key_here

# Grok Model Configuration
GROK_MODEL=grok-4-fast-non-reasoning

# Agent Configuration
TEMPERATURE=0.3
MAX_TOKENS=2000

# Thread Management
THREAD_ID=test
```

## Usage

### Fetch Live NBA Data

**List today's games:**
```bash
python main.py --list-games
```

**Generate commentary for a team's most recent game:**
```bash
python main.py --team LAL
```

**Generate commentary for a specific game:**
```bash
python main.py --game-id 0022400123
```

**Generate commentary for the first 10 minutes of a game:**
```bash
python main.py --game-id 0022400123 --duration 10
```

**Generate commentary for the first 25 minutes (first half minus last minute):**
```bash
python main.py --team LAL --duration 25
```

**Fetch live game data (for ongoing games):**
```bash
python main.py --team GSW --live
```

**Generate commentary in Spanish:**
```bash
python main.py --game-id 0022400350 --duration 10 --language es
```

**Generate commentary in French:**
```bash
python main.py --team LAL --duration 5 --language fr
```

### Manual Input

Process a single play-by-play event:

```bash
echo '{"event_type": "shot", "description": "LeBron James makes 3-pointer", "player": "LeBron James", "team": "Lakers", "score": "LAL 105-98 MIA", "time": "Q4 2:30"}' | python main.py
```

### Process Multiple Events

Create a JSON file with play-by-play events:

```json
[
  {
    "event_type": "shot",
    "description": "LeBron James makes 3-pointer",
    "player": "LeBron James",
    "team": "Lakers",
    "score": "LAL 105-98 MIA",
    "time": "Q4 2:30",
    "quarter": 4
  },
  {
    "event_type": "foul",
    "description": "Personal foul on Anthony Davis",
    "player": "Anthony Davis",
    "team": "Lakers",
    "score": "LAL 105-98 MIA",
    "time": "Q4 2:15",
    "quarter": 4
  }
]
```

Process the file:

```bash
python main.py --input events.json --output commentary.json
```

### TTS Format Output

Get output formatted for text-to-speech:

```bash
python main.py --input events.json --output commentary.txt --format tts
```

### Using as a Python Module

```python
from agent import NBACommentaryAgent

agent = NBACommentaryAgent()

event = {
    "event_type": "shot",
    "description": "Steph Curry makes 3-pointer from 28 feet",
    "player": "Steph Curry",
    "team": "Warriors",
    "score": "GSW 98-95 BOS",
    "time": "Q4 1:45",
    "quarter": 4
}

commentary = agent.process_event(event)
print(commentary.commentators[0].commentary)
```

## Input Format

Play-by-play events should be dictionaries with the following optional fields:

- `event_type`: Type of event (e.g., "shot", "foul", "turnover", "timeout")
- `description`: Human-readable description of the event
- `player`: Player name involved
- `team`: Team name
- `score`: Current score
- `time`: Game time (e.g., "Q4 2:30")
- `quarter`: Quarter number (1-4)
- Any additional fields will be included in the prompt

## Output Format

The agent returns a `CommentaryOutput` object with:

- `commentators`: List of 1-2 commentators, each with:
  - `name`: Commentator identifier
  - `commentary`: Text to be spoken
  - `tone`: Optional tone/style indicator
- `game_context`: Optional additional game context

## Configuration

Edit `.env` or set environment variables:

- `XAI_API_KEY`: Your xAI API key (required)
- `GROK_MODEL`: Model name (default: "grok-beta")
- `TEMPERATURE`: LLM temperature (default: 0.7)
- `MAX_TOKENS`: Maximum tokens (default: 2000)
- `THREAD_ID`: Thread identifier for context (default: "nba_game_commentary")

## Project Structure

```
xai-hack/
├── main.py           # Entrypoint with CLI
├── agent.py          # Langchain agent with Grok integration
├── models.py         # Pydantic models for structured output
├── nba_data.py       # NBA API integration for fetching play-by-play data
├── config.py         # Configuration management
├── requirements.txt  # Dependencies
├── README.md         # This file
└── .env              # Environment variables (create this)
```

## Development

The codebase is designed to be simple and maintainable:

- **Single thread management**: All events use the same conversation thread for context
- **Structured output**: Pydantic models ensure type safety and validation
- **Error handling**: Graceful error handling with informative messages
- **Extensible**: Easy to add new features or modify behavior

## License

MIT
