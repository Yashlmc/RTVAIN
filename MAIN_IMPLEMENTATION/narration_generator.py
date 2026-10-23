import os
import json
from groq import Groq
from dotenv import load_dotenv


class CommentaryGenerator:
    """
    Takes all events from one Context Engine sequence (5 frames or so)
    and generates ONE short natural commentary line using Groq LLM.
    """

    def __init__(self, model="llama-3.1-8b-instant"):
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")

        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_sequence_commentary(self, sequence_events, match_context):
        """
        sequence_events: list of dicts
            e.g. [{"event": "kick", "frame": 55}, {"event": "goal", "direction": "top-right", "frame": 60}]
        match_context: dict
            e.g. {"player_kicker": "Messi", "goalkeeper": "Neuer", "score": "2 - 2", "penalty_no": 5}
        """

        if not sequence_events:
            return None

        # Compact JSON string for events
        event_json = json.dumps(sequence_events, indent=2)

        prompt = f"""
        You are a live football commentator.
        Based on the following football penalty sequence and match context, generate one short, natural-sounding commentary line.
        The line should sound live, emotional, and professional — as if describing the action that just happened.

        Match context before taking penalty:
        - Kicker: {match_context.get("player_kicker")}
        - Goalkeeper: {match_context.get("goalkeeper")}
        - Score: {match_context.get("score")}

        Penalty sequence (events):
        {event_json}

        Guidelines:
        - Generate ONE single sentence only (which can be spoken in ~5 seconds).
        - Reflect the full action (kick → goal/miss/save).
        - Mention players' names and direction (if given).
        - No repetition, no extra commentary lines.
        - Keep it concise, energetic, and realistic.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional sports commentator narrating a live football match."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=50,
            )

            commentary = response.choices[0].message.content.strip()
            return commentary

        except Exception as e:
            print(f"Groq API error: {e}")
            return None



if __name__ == "__main__":
    # Create instance (auto-loads key from .env)
    generator = CommentaryGenerator()

    # Context from match state (this can come from UI or match engine)
    match_context = {
        "player_kicker": "Messi",
        "goalkeeper": "Neuer",
        "score": "3 - 2"
    }

    # Example events from context engine (for one full play)
    sequence_events = [
        {"event": "run_up", "frame": 48},
        {"event": "kick", "frame": 55},
        {"event": "keeper_dive", "direction": "top-left", "frame": 57},
        {"event": "goal", "direction": "top-right", "frame": 60}
    ]

    commentary = generator.generate_sequence_commentary(sequence_events, match_context)

    print("\n[Sequence Events]:", sequence_events)
    print("[Generated Commentary]:", commentary)
