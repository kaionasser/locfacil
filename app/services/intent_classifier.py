import re


INTENT_PATTERNS = {
    "greeting": [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b", r"\bgood afternoon\b"
    ],
    "price_query": [
        r"\bprice\b", r"\bcost\b", r"\bhow much\b", r"\brental rate\b", r"\bcharge\b"
    ],
    "availability_query": [
        r"\bavailable\b", r"\bavailability\b", r"\bin stock\b", r"\bdo you have\b"
    ],
    "customer_info_query": [
        r"\bcustomer\b", r"\bclient\b", r"\bcontact\b", r"\bphone\b", r"\bemail\b"
    ],
    "return_date_query": [
        r"\breturn date\b", r"\bwhen.*return\b", r"\bdue date\b", r"\bwhen is .* due\b"
    ],
    "policy_query": [
        r"\bpolicy\b", r"\bdeposit\b", r"\bcancellation\b", r"\blate return\b", r"\brules\b"
    ]
}


def classify_intent(message: str) -> str:
    text = message.strip().lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent

    return "unknown"


if __name__ == "__main__":
    test_messages = [
        "Hi there",
        "How much is the concrete mixer?",
        "Is the backhoe loader available tomorrow?",
        "Show me the customer phone for Maria Silva",
        "When is John Smith supposed to return the scaffolding?",
        "What is your deposit policy?",
        "I want something else"
    ]

    for msg in test_messages:
        print(f"Message: {msg}")
        print(f"Intent: {classify_intent(msg)}")
        print("-" * 40)