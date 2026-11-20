import os
import requests
from dotenv import load_dotenv

load_dotenv()

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://localhost:8080")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME", "Loc Facil")


def send_whatsapp_message(to: str, message: str) -> bool:
    """Send a WhatsApp message via Evolution API."""
    url = f"{EVOLUTION_API_URL}/message/sendText/{INSTANCE_NAME}"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "number": to,
        "text": message,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"[WhatsApp] Message sent to {to}", flush=True)
        return True
    except requests.RequestException as e:
        print(f"[WhatsApp] Failed to send message to {to}: {e}", flush=True)
        return False