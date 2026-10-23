# import json
# import time
# from groq import Groq


# class LiveCommentaryGenerator:
#     """
#     Converts structured context engine outputs into human-like live commentary text
#     using Groq's LLM API.
#     """

#     def __init__(self, api_key, model="llama-3.1-8b-instant"):
#         self.client = Groq(api_key=api_key)
#         self.model = model
#         self.last_event = None

#     def generate_commentary(self, event, context):
#         """
#         Generate a natural-sounding commentary line from an event and context.
#         event: dict, e.g. {"event":"goal", "direction":"top-right", "frame":60}
#         context: dict, e.g. {"player_kicker":"Ronaldo","goalkeeper":"Buffon","score":"2-2","penalty_no":5}
#         """

#         if not event or event == self.last_event:
#             return None  # Skip duplicates or empty

#         # Build structured prompt
#         prompt = f"""
#         You are a live football commentator. 
#         Based on the following event and context, generate one or two lines of lively, natural-sounding football commentary.

#         Event data (JSON):
#         {json.dumps(event, indent=2)}

#         Match context:
#         - Kicker: {context.get("player_kicker")}
#         - Goalkeeper: {context.get("goalkeeper")}
#         - Current Score: {context.get("score")}
#         - Penalty No: {context.get("penalty_no")}

#         Commentary style:
#         - Real-time reaction (as if happening now)
#         - Energetic, descriptive, and professional tone
#         - Avoid repeating old events
#         - Mention players’ names and direction if available
#         - Keep it short (1–2 sentences)
#         """

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a professional football commentator providing live narration."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.9,
#                 max_tokens=80,
#             )

#             commentary = response.choices[0].message.content.strip()
#             self.last_event = event
#             return commentary

#         except Exception as e:
#             print(f"Groq API error: {e}")
#             return None


# # ---------------- Example Demo ----------------

# if __name__ == "__main__":
#     # Initialize with your Groq API key
#     generator = LiveCommentaryGenerator(api_key="your-hardcoded-api-key")

#     # Example context (can be dynamically updated in real match)
#     game_context = {
#         "player_kicker": "Cristiano Ronaldo",
#         "goalkeeper": "Manuel Neuer",
#         "score": "1 - 1",
#         "penalty_no": 3
#     }

#     # Example simulated stream of context engine outputs
#     simulated_events = [
#         {"event": "kick", "frame": 55},
#         {"event": "goal", "direction": "top-right", "frame": 60},
#         {"event": "save", "direction": "bottom-left", "frame": 85},
#     ]

#     for e in simulated_events:
#         print(f"\n[Context Engine Output] {e}")
#         commentary = generator.generate_commentary(e, game_context)
#         if commentary:
#             print(f"[Live Commentary] {commentary}")
#         time.sleep(1.5)  # simulate live pacing



###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


import json
from groq import Groq


class SequenceCommentaryGenerator:
    """
    Takes all events from one Context Engine sequence (5 frames or so)
    and generates ONE short natural commentary line using Groq LLM.
    """

    def __init__(self, api_key, model="llama-3.1-8b-instant"):
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
                    {"role": "system", "content": "You are a professional sports commentator narrating a live football match."},
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


# ---------------- Example Demo ----------------

if __name__ == "__main__":
    generator = SequenceCommentaryGenerator(api_key="your-hardcoded-api-key")

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
