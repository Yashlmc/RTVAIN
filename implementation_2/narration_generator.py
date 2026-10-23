import json
from groq import Groq

class PenaltyCommentaryGenerator:
    """
    Converts a sequence of football penalty events into ONE natural-sounding commentary line.
    """

    def __init__(self, api_key, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_commentary(self, sequence_events, match_context):
        """
        sequence_events: list of dicts, e.g.
            [{"event":"run_up","frame":48}, {"event":"kick","frame":55}, {"event":"goal","direction":"top-right","frame":60}]
        match_context: dict, e.g.
            {"player_kicker": "Messi", "goalkeeper": "Neuer", "score": "3 - 2", "penalty_no": 5}
        """
        if not sequence_events:
            return "No major events detected."

        # Convert events to JSON string
        events_json = json.dumps(sequence_events, indent=2)

        prompt = f"""
        You are a live football commentator.
        Based on the following football penalty sequence and match context, generate ONE short, natural-sounding commentary line.
        The line should sound live, emotional, and professional — as if describing the action that just happened.

        Match context before penalty:
        - Kicker: {match_context.get("player_kicker")}
        - Goalkeeper: {match_context.get("goalkeeper")}
        - Score: {match_context.get("score")}
        - Penalty no: {match_context.get("penalty_no")}

        Penalty sequence (events):
        {events_json}

        Guidelines:
        - Generate ONE single sentence only (~5 seconds speech).
        - Mention players' names and directions (if present).
        - Reflect full action (waiting/run-up/kick → ball_in_air → goal/save/miss).
        - Keep concise, energetic, and realistic.
        - No repetition, no extra lines.
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
    generator = PenaltyCommentaryGenerator(api_key="your-hardcoded-api-key")

    # Example match context
    match_context = {
        "player_kicker": "Ronaldo",
        "goalkeeper": "Kahn",
        "score": "0-0",
        "penalty_no": 2
    }

    # Example events from sliding window engine
    sequence_events = [
        {'event': 'kick', 'ball_dir': [5, -5], 'keeper_dir': [0, 0], 'dist_kicker_ball': 5.0, 'dist_ball_goal': 209.4, 'dist_ball_keeper': 161.4},
        {'event': 'ball_in_air', 'ball_dir': [110, -5], 'keeper_dir': [0, -5], 'dist_kicker_ball': 167.63, 'dist_ball_goal': 65.76, 'dist_ball_keeper': 7.07},
        {'event': 'miss', 'ball_dir': [60, 180], 'keeper_dir': [0, 0], 'dist_kicker_ball': 264.05, 'dist_ball_goal': 215.41, 'dist_ball_keeper': 161.01}
        ]


    commentary = generator.generate_commentary(sequence_events, match_context)

    print("\n[Sequence Events]:", sequence_events)
    print("[Generated Commentary]:", commentary)


###################################################################################################
###################################################################################################

"""
# Example Output: 
# " Messi steps up, runs towards Neuer, and unleashes a powerful shot, but unfortunately, the Argentine maestro's attempt is met with a brilliant save from the Bavarian wall!"
# "AND WHAT A MISS BY RONALDO! THE BALL SOARS INTO THE AIR, BUT BUFFON LEANS BACK AND TIPS IT OVER THE CROSSBAR!"
# "Chhetri's fifth penalty attempt flies towards the top corner, but Singh leaps to his right, and it's a magnificent save, preserving the 2-2 stalemate!
"""