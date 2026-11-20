from app.services.intent_classifier import classify_intent
from app.services.retriever import (
    find_product_by_name,
    find_customer_by_name,
    find_rentals_by_customer,
    find_policy_by_type,
)
from app.services.rag_service import generate_grounded_response


KNOWN_PRODUCTS = [
    "concrete mixer",
    "frame scaffolding set",
    "plate compactor",
    "backhoe loader",
    "mini backhoe loader",
    "excavator",
    "small dump truck",
    "office container",
    "office container (20 ft)",
]

KNOWN_POLICY_KEYWORDS = [
    "deposit",
    "cancellation",
    "late return",
    "heavy equipment",
    "new customer",
]

KNOWN_CUSTOMERS = [
    "Maria Silva",
    "John Smith",
    "Ana Costa",
    "Carlos Mendes",
    "Emily Johnson",
    "Robert Brown",
]


def extract_product_name(message: str):
    text = message.lower()
    for product in sorted(KNOWN_PRODUCTS, key=len, reverse=True):
        if product in text:
            return product
    return None


def extract_customer_name(message: str):
    text = message.lower()
    for name in KNOWN_CUSTOMERS:
        if name.lower() in text:
            return name
    return None


def extract_policy_keyword(message: str):
    text = message.lower()
    for keyword in KNOWN_POLICY_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def build_response(message: str):
    intent = classify_intent(message)
    retrieved_data = None

    if intent == "greeting":
        return {
            "intent": intent,
            "data": None,
            "response": "Hello! How can I help you with rentals today?"
        }

    if intent in ["price_query", "availability_query"]:
        product_name = extract_product_name(message)
        if not product_name:
            return {
                "intent": intent,
                "data": None,
                "response": "I could not identify the product name in your message."
            }
        retrieved_data = find_product_by_name(product_name)

    elif intent == "customer_info_query":
        customer_name = extract_customer_name(message)
        if not customer_name:
            return {
                "intent": intent,
                "data": None,
                "response": "I could not identify the customer name in your message."
            }
        retrieved_data = find_customer_by_name(customer_name)

    elif intent == "return_date_query":
        customer_name = extract_customer_name(message)
        if not customer_name:
            return {
                "intent": intent,
                "data": None,
                "response": "I could not identify the customer name in your message."
            }
        retrieved_data = find_rentals_by_customer(customer_name)

    elif intent == "policy_query":
        policy_keyword = extract_policy_keyword(message)
        if not policy_keyword:
            return {
                "intent": intent,
                "data": None,
                "response": "I could not identify which policy you are asking about."
            }
        retrieved_data = find_policy_by_type(policy_keyword)

    else:
        return {
            "intent": "unknown",
            "data": None,
            "response": "Sorry, I could not understand your request."
        }

    if not retrieved_data:
        return {
            "intent": intent,
            "data": [],
            "response": "I could not find relevant information in the database."
        }

    final_response = generate_grounded_response(
        user_message=message,
        intent=intent,
        retrieved_data=retrieved_data
    )

    return {
        "intent": intent,
        "data": retrieved_data,
        "response": final_response
    }


if __name__ == "__main__":
    test_messages = [
        "How much is the concrete mixer?"
    ]

    for msg in test_messages:
        print(f"\nMessage: {msg}")
        result = build_response(msg)
        print("Intent:", result["intent"])
        print("Response:", result["response"])
        print("Data:", result["data"])
        print("-" * 50)