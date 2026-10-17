"""Offline GenAI chatbot using a small pre-trained model (SmolLM2-360M-Instruct).

Run:  python chatbot.py
Commands inside the chat:  /reset  clears history,  /exit  stops the chatbot
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

SYSTEM_PROMPT = """
You are a helpful AI teaching assistant.
Explain concepts in simple English.
Use examples whenever helpful.
If you are unsure about an answer, say so.
If you do not know about a person, company, product or recent event, say "I don't know". Never make up facts.
"""

MAX_NEW_TOKENS = 120        # maximum length of each answer
MAX_HISTORY_MESSAGES = 6    # how many recent messages the bot remembers

torch.set_num_threads(4)

print("Loading model (first run downloads it, later runs use the saved copy)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to("cpu")
model.eval()
print("Model loaded. Running on CPU.")


def generate_response(user_message, history=None):
    """Return the chatbot's answer to user_message, using recent history."""
    if history is None:
        history = []

    recent_history = history[-MAX_HISTORY_MESSAGES:]

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + recent_history
        + [{"role": "user", "content": user_message}]
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Keep only the newly generated tokens (not the prompt)
    input_length = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(
        output_ids[0][input_length:], skip_special_tokens=True
    ).strip()

    return answer or "I could not generate a response. Please try again."


def start_chat():
    history = []

    print("=" * 50)
    print("       OFFLINE GENAI CHATBOT")
    print("=" * 50)
    print("Ask any question.")
    print("Type /reset to clear history.")
    print("Type /exit to stop.")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "/exit":
            print("Bot: Goodbye!")
            break

        if question.lower() == "/reset":
            history.clear()
            print("Bot: Conversation history cleared.")
            continue

        print("Bot: Thinking...")
        answer = generate_response(question, history)
        print("\nBot:", answer)

        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        # Keep the history small
        if len(history) > 12:
            history = history[-12:]


if __name__ == "__main__":
    start_chat()
