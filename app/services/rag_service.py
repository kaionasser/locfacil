import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_grounded_response(user_message: str, intent: str, retrieved_data):
    context = json.dumps(retrieved_data, indent=2, default=str)

    system_prompt = """
You are a rental business AI assistant.

Your job is to answer the user only using the provided business data.
Do not invent information.
If the data is missing or unclear, say that clearly.
Be short, professional, and helpful.
"""

    user_prompt = f"""
User message:
{user_message}

Detected intent:
{intent}

Retrieved business data:
{context}

Instructions:
- Answer only based on the retrieved business data.
- If the user asks about price, summarize the available rates clearly.
- If the user asks about availability, use the available quantity if present.
- If the user asks about customer info, summarize the customer details.
- If the user asks about return dates, mention the relevant rental records.
- If the user asks about a policy, summarize the policy clearly.
- If there is no useful data, say you could not find the requested information.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text


if __name__ == "__main__":
    sample_data = [
        {
            "Name": "Concrete Mixer",
            "PricePerDay": 55.0,
            "PricePerWeek": 180.0,
            "PricePerMonth": 520.0,
            "Description": "Portable mixer for small and medium jobs"
        }
    ]

    result = generate_grounded_response(
        user_message="How much is the concrete mixer?",
        intent="price_query",
        retrieved_data=sample_data
    )

    print(result)