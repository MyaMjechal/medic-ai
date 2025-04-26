import gradio as gr
import asyncio
from emotion_chat_2 import EmotionChatbot

# Initialize the EmotionChatbot instance
chatbot = EmotionChatbot()

async def chat_async(message, history):
    # Send the message to the chatbot and get a response
    response = await chatbot.chat(message)
    # Append the user message and bot response to history
    history = history or []
    history.append((message, response))
    return history


def respond(message, history):
    # Run the async chat in a synchronous context
    return asyncio.run(chat_async(message, history)), history

# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Emotion-Aware Chatbot\nInteract with the EmotionChatbot. Previous messages are shown below.")
    chat_ui = gr.Chatbot(label="Chat History")
    state = gr.State([])
    user_input = gr.Textbox(
        placeholder="Type your message here...",
        label="You",
        lines=1,
        interactive=True
    )
    # On Enter or Submit, send to respond, update chat_ui and state
    user_input.submit(respond, [user_input, state], [chat_ui, state])
    demo.launch(share=True)
