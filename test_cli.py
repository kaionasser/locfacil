#!/usr/bin/env python3
"""
CLI Test Mode for LocFacil AI Assistant
Run this to test the assistant without WhatsApp.
Usage: python test_cli.py
"""

from app.services.tool_calling_assistant import run_assistant

def main():
    print("=" * 50)
    print("  LocFacil AI Assistant - CLI Test Mode")
    print("=" * 50)
    print("Type your message and press Enter to chat.")
    print("Type 'quit' or 'exit' to stop.\n")

    session_id = "cli_test_user"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            print("Assistant: ", end="", flush=True)
            reply = run_assistant(session_id, user_input)
            print(reply)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    main()