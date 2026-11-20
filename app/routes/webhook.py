from fastapi import APIRouter, Request
from app.services.whatsapp_service import send_whatsapp_message
from app.services.tool_calling_assistant import run_assistant

import os
from dotenv import load_dotenv

load_dotenv()

BOT_NUMBER = os.getenv("BOT_NUMBER", "18503418161")
router = APIRouter()

BOT_ENABLED = False # Global flag to enable/disable bot responses
processed_ids = set()  # tracks message IDs already handled


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/webhook/evolution")
async def evolution_webhook(request: Request):
    try:
        payload = await request.json()
        event = payload.get("event", "")
        
        if not BOT_ENABLED:        # 👈 add this
            return {"ok": True}

        if event != "messages.upsert":
            return {"ok": True}

        data = payload.get("data", {})

        if not isinstance(data, dict):
            return {"ok": True}

        key = data.get("key", {})

        if key.get("fromMe"):
            return {"ok": True}

        sender = key.get("remoteJid", "")
        if BOT_NUMBER in sender:
            return {"ok": True}

        # Deduplicate by message ID
        message_id = key.get("id")
        if message_id in processed_ids:
            print(f"[Webhook] Duplicate message {message_id}, skipping", flush=True)
            return {"ok": True}
        processed_ids.add(message_id)

        message_text = (
            data.get("message", {}).get("conversation")
            or data.get("message", {}).get("extendedTextMessage", {}).get("text")
        )

        if not message_text:
            return {"ok": True}

        print(f"[Webhook] Message from {sender}: {message_text}", flush=True)

        ai_reply = run_assistant(sender, message_text)
        print(f"[Webhook] AI reply: {ai_reply}", flush=True)
        send_whatsapp_message(sender, ai_reply)

    except Exception as e:
        print(f"[Webhook] Error: {e}", flush=True)

    return {"ok": True}