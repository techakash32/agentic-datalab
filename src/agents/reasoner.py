import os
from openai import OpenAI

class ReasonerAgent:
    def __init__(self):
        # First, try to get the API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set it as an environment variable or add your key in .env for local testing."
            )
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)

    def generate_report(self, eval_json):
        prompt = f"Summarize this evaluation JSON in clear language:\n{eval_json}"

        # Call the OpenAI chat completion API
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        return response.choices[0].message.content

