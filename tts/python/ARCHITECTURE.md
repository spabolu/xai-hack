# GrokCast NBA Architecture

## How It Works

```mermaid
flowchart LR
    subgraph Input
        A[📺 NBA Game Data]
        B[🏀 Your Team]
        C[🌍 Language]
    end

    subgraph "Grok AI (xAI)"
        D[🧠 Grok 4.1\nCommentary]
        E[🔍 X Search\nPlayer Stats]
    end

    subgraph Output
        F[🎙️ Grok Voice\nTTS]
        G[🔊 Live Audio]
    end

    A --> D
    B --> D
    C --> D
    D -->|tokens| F
    E -->|filler| F
    F -->|stream| G
```

## Simple Flow

```mermaid
flowchart TD
    A[Game Event Happens] --> B{Is TTS busy?}
    B -->|No| C[Generate Commentary]
    B -->|Yes| D[Interrupt Current Speech]
    D --> C
    C --> E[Stream to Voice]
    E --> F[Play Audio]
    
    G[No Event?] --> H[Search Player Stats]
    H --> E
```

## Real-Time Pipeline

```
NBA Play-by-Play  →  Grok 4.1 LLM  →  Grok Voice TTS  →  Speaker
     (JSON)           (tokens)         (WebSocket)       (audio)
                          ↓
                    X Search fills
                    quiet moments
```
