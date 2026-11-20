import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

from app.services.retriever import (
    find_product_by_name,
    find_customer_by_name,
    find_rentals_by_customer,
    find_policy_by_type,
    get_all_products,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Memory: stores last 10 messages per sender phone number
conversation_history = defaultdict(list)
MAX_HISTORY = 10

TOOLS = [
    {
        "type": "function",
        "name": "get_all_products",
        "description": "Get the full list of rental products and equipment offered by the business.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "find_product_by_name",
        "description": "Find a rental product or equipment by name mentioned by the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The product or equipment name mentioned by the user."
                }
            },
            "required": ["product_name"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "find_customer_by_name",
        "description": "Find customer details by customer name.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The customer name mentioned by the user."
                }
            },
            "required": ["customer_name"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "find_rentals_by_customer",
        "description": "Find rental records for a given customer, including return dates and rental status.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "The customer name mentioned by the user."
                }
            },
            "required": ["customer_name"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "find_policy_by_type",
        "description": "Find a business policy by topic such as deposit, cancellation, late return, or new customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "description": "The policy topic the user is asking about."
                }
            },
            "required": ["policy_type"],
            "additionalProperties": False
        },
        "strict": True
    }
]

TOOL_FUNCTIONS = {
    "get_all_products": lambda **kwargs: get_all_products(),
    "find_product_by_name": lambda **kwargs: find_product_by_name(kwargs["product_name"]),
    "find_customer_by_name": lambda **kwargs: find_customer_by_name(kwargs["customer_name"]),
    "find_rentals_by_customer": lambda **kwargs: find_rentals_by_customer(kwargs["customer_name"]),
    "find_policy_by_type": lambda **kwargs: find_policy_by_type(kwargs["policy_type"]),
}

SYSTEM_PROMPT = """
You are LocFacil's AI rental assistant.

Use the available tools whenever the user asks about products, equipment, pricing, customers, rentals, return dates, or policies.
Only answer using the tool results when business data is needed.
Do not invent information.

Important rules:
- Do not invent customer details, return dates, prices, or policies.
- If the tool results are empty or unclear, say that clearly.
- If the user asks whether the business offers a product, you may confirm that the product exists in the catalog.
- Only mention missing or unconfirmed availability if the user specifically asks about current availability, quantity available now, or immediate scheduling.

Style rules:
- Keep answers short, human, and natural.
- Prefer WhatsApp-style responses: clear, friendly, and easy to read.
- Avoid long paragraphs.
- When mentioning prices, dates, or options, format them as short bullet-style lines using hyphens.
- Use simple wording.
- Use contractions and natural chat wording when appropriate.
- Be helpful and positive, but never pushy, aggressive, or misleading.
- If appropriate, end with a short follow-up question.
- Do not add unnecessary explanations.

Response format:
- Maximum around 3 to 6 short lines for most answers.
- Use list format for prices, rental options, policies, and returned data when helpful.
- Sound like a real assistant helping a customer in chat.
"""


def run_assistant(sender: str, user_message: str) -> str:
    # Add new user message to this sender's history
    history = conversation_history[sender]
    history.append({"role": "user", "content": user_message})

    # Keep only the last MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        conversation_history[sender] = history

    # Build the input: history becomes the conversation
    first_response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=SYSTEM_PROMPT,
        input=history,
        tools=TOOLS
    )

    function_calls = [
        item for item in first_response.output
        if item.type == "function_call"
    ]

    if not function_calls:
        ai_reply = first_response.output_text
        history.append({"role": "assistant", "content": ai_reply})
        return ai_reply

    tool_outputs = []
    for call in function_calls:
        tool_name = call.name
        arguments = json.loads(call.arguments)
        if tool_name not in TOOL_FUNCTIONS:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            result = TOOL_FUNCTIONS[tool_name](**arguments)
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, default=str)
        })

    final_response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=SYSTEM_PROMPT,
        previous_response_id=first_response.id,
        input=tool_outputs
    )

    ai_reply = final_response.output_text
    history.append({"role": "assistant", "content": ai_reply})
    return ai_reply