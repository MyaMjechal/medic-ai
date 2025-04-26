# app.py
import gradio as gr
import asyncio
from emotion_chat import EmotionChatbot  # adjust import to wherever your class lives

# 1. Initialize your chatbot (it already calls nest_asyncio.apply())
chatbot = EmotionChatbot()

def respond(user_message: str) -> str:
    """
    Synchronous wrapper that runs the async `chat` coroutine
    on the existing event loop and returns its string result.
    """
    # If your class method is named something else, adjust below:
    coro = chatbot.chat(user_message)
    return asyncio.get_event_loop().run_until_complete(coro)

iface = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here…", label="You"),
    outputs=gr.Textbox(label="Emotion-Aware Chatbot"),
    title="Emotion-Aware Chatbot",
    description="An empathetic, emotion‑aware assistant powered by Mistral-7B Instruct."
)

if __name__ == "__main__":
    iface.launch(share=True)
