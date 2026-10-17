# 🤖 Offline GenAI Chatbot

A simple chatbot that runs fully on my own computer (CPU only, no API key, no internet needed after the first download). It uses a small pre-trained language model, **SmolLM2-360M-Instruct** by Hugging Face, and I built the chatbot around it.

## What it does
- Answers questions in simple English like a teaching assistant
- Remembers the last few messages, so follow-up questions work (for example: "What is Machine Learning?" → "Give me a real-life example of it.")
- `/reset` clears the memory, `/exit` stops the chatbot

## How it works
1. Loads the SmolLM2 model and tokenizer with Hugging Face Transformers and PyTorch
2. A **system prompt** tells the model how to behave
3. The last 6 messages and the new question are turned into a chat prompt
4. The model writes an answer (max 120 new tokens, `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.1`)
5. Only the new answer is shown, and the history is updated

## Tech used
Python · PyTorch · Hugging Face Transformers · Jupyter Notebook

## How to run

```bash
pip install -r requirements.txt
python chatbot.py
```

The first run downloads the model (a few hundred MB). After that it is saved on your computer.

You can also open `AI_Project.ipynb` and run the cells one by one.

## Sample chat
<!-- Paste 2-3 real questions and answers from your own run here -->

```
You: What is Machine Learning?
Bot: ...
```

## Limitations
- It is a small model (360M parameters), so it can make mistakes or give short answers
- It can make up facts about people, companies and recent topics it was not trained on (for example, it described "Claude AI" as a French artist/singer). Use it for simple concept explanations, not for facts
- It runs on CPU, so answers take a few seconds

## Credits
Model: [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (pre-trained, not trained by me).
