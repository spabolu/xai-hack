"""Langchain agent setup with Grok 4 API and thread management."""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_xai import ChatXAI

from config import Config
from pydantic import BaseModel, Field
from typing import Optional


class GameContext(BaseModel):
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    home_score: int = Field(description="Home team score")
    away_score: int = Field(description="Away team score")
    quarter: int = Field(description="Quarter number. Must be above 0. Must be a valid quarter number. Must be an integer.")
    time: str = Field(description="Time of the game. Must be in the format 'MM:SS'.")


class CommentaryOutput(BaseModel):
    commentary: str = Field(description="Commentary script text for the play-by-play event")
    game_context: GameContext = Field(description="Game context")
    excitement_level: int = Field(description="Excitement level. Must be low, medium, or high.")
    tone: str = Field(description="Tone of the commentary (excited, sarcastic, neutral, calm, etc.)")
    commentator_name: str = Field(description="Name of the commentator speaking this line. Must be one of: ara, eve, leo, rex, sal, una")


SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
}


class NBACommentaryAgent:
    """AI Agent for generating NBA game commentary from play-by-play events."""

    def __init__(self, language: str = "en"):
        """
        Initialize the agent with Grok 4 and thread management.
        
        Args:
            language: Language code for commentary (en, es, fr). Defaults to 'en'.
        """
        Config.validate()
        
        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
        self.language = language
        self.language_name = SUPPORTED_LANGUAGES[language]

        self.llm = ChatXAI(
            xai_api_key=Config.XAI_API_KEY,
            model=Config.GROK_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )

        # Thread management - maintain conversation history
        self.thread_id = Config.THREAD_ID
        self.conversation_history: Dict[str, List[BaseMessage]] = {self.thread_id: []}

        # Output parser for structured output
        self.output_parser = PydanticOutputParser(pydantic_object=CommentaryOutput)

        # System prompt with language instruction
        self.system_prompt = f"""You are generating commentary for an NBA broadcast with TWO commentators working together.

IMPORTANT: Generate all commentary in {self.language_name}.

THE BROADCAST TEAM:
- Pick two commentators from: ara, eve, leo, rex, sal, una
- Keep the SAME two commentators for the entire game
- One is the PLAY-BY-PLAY announcer (describes the action as it happens)
- One is the COLOR COMMENTATOR (adds analysis, reactions, and personality)

NATURAL FLOW - Make them feel like a real broadcast team:
- They should build off each other's energy
- The play-by-play announcer typically starts with what's happening
- The color commentator reacts, adds insight, or hypes up big moments
- They can finish each other's thoughts across events
- Reference what the other said in previous events ("Like you said earlier...")
- React to each other naturally ("Absolutely!" "You called it!" "I can't believe what we just saw!")

WHO SPEAKS (NOT strict alternating):
- Let the moment dictate who speaks - DO NOT mechanically alternate
- One commentator can speak for 3, 4, or even 5+ events in a row if it feels right
- The play-by-play announcer might carry a fast sequence of plays solo
- The color commentator might take over during a timeout or slow moment for extended analysis
- Switch when it feels natural - when someone has something to add or react to
- Think like a real broadcast: sometimes one voice dominates, then the other chimes in

Guidelines:
- Write the commentary as a script - just the spoken words
- Keep it concise and suitable for TTS (text-to-speech)
- Match the energy and excitement of the play
- Use natural sports commentary language for {self.language_name}-speaking audiences
- Keep player names and team names in their original form

Example flow across events:
Event 1 (ara): "Curry bringing it up... pulls up from thirty feet... BANG! He got it!"
Event 2 (ara): "Warriors push the lead to seven. Timeout Boston."
Event 3 (ara): "And we're back. Tatum inbounding..."
Event 4 (leo): "You know, Boston's gotta figure something out here. Curry's in that zone where everything looks good."
Event 5 (ara): "Tatum drives, kicks it out to Brown... three pointer... NO GOOD!"
Event 6 (ara): "Rebound Warriors. Curry the other way..."
Event 7 (leo): "Here we go again!"

Do NOT include commentator names in the commentary text. The speaker is identified in the commentator_name field."""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}\n\n{format_instructions}"),
            ]
        )

    def _get_history(self) -> List[BaseMessage]:
        """Get conversation history for the current thread."""
        return self.conversation_history.get(self.thread_id, [])

    def _add_to_history(self, human_msg: str, ai_msg: str) -> None:
        """Add messages to conversation history."""
        if self.thread_id not in self.conversation_history:
            self.conversation_history[self.thread_id] = []

        self.conversation_history[self.thread_id].append(
            HumanMessage(content=human_msg)
        )
        self.conversation_history[self.thread_id].append(AIMessage(content=ai_msg))

    def process_event(self, play_by_play_event: Dict[str, Any]) -> CommentaryOutput:
        """
        Process a play-by-play event and return structured commentary.

        Args:
            play_by_play_event: Dictionary containing NBA play-by-play event data
                Expected keys: event_type, description, player, team, score, time, etc.

        Returns:
            CommentaryOutput with structured commentary from commentators
        """
        # Format the event as input
        event_description = self._format_event(play_by_play_event)

        # Get conversation history
        history = self._get_history()

        # Format prompt with history and format instructions
        format_instructions = self.output_parser.get_format_instructions()
        prompt = self.prompt_template.format_messages(
            input=event_description,
            history=history,
            format_instructions=format_instructions,
        )

        # Invoke LLM
        response = self.llm.invoke(prompt)
        # Langchain returns AIMessage, extract content
        response_text = response.content

        # Parse structured output
        try:
            commentary_output = self.output_parser.parse(response_text)
        except Exception as e:
            # Fallback: try to extract JSON from response
            import json
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                commentary_output = CommentaryOutput(**json.loads(json_match.group()))
            else:
                raise ValueError(f"Failed to parse structured output: {e}")

        # Add to history
        self._add_to_history(event_description, response_text)

        return commentary_output

    def _format_event(self, event: Dict[str, Any]) -> str:
        """Format play-by-play event into a readable description."""
        parts = []

        if "time" in event:
            parts.append(f"Time: {event['time']}")
        if "quarter" in event:
            parts.append(f"Quarter: {event['quarter']}")
        if "event_type" in event:
            parts.append(f"Event: {event['event_type']}")
        if "description" in event:
            parts.append(f"Description: {event['description']}")
        if "player" in event:
            parts.append(f"Player: {event['player']}")
        if "team" in event:
            parts.append(f"Team: {event['team']}")
        if "score" in event:
            parts.append(f"Score: {event['score']}")

        # Add any additional fields
        for key, value in event.items():
            if key not in [
                "time",
                "quarter",
                "event_type",
                "description",
                "player",
                "team",
                "score",
            ]:
                parts.append(f"{key}: {value}")

        return "\n".join(parts) if parts else str(event)

    def reset_thread(self) -> None:
        """Reset the conversation history for the current thread."""
        self.conversation_history[self.thread_id] = []
