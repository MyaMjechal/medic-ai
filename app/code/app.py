import dash
import os
import asyncio
from dash import html, dcc, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from emotion_chat import EmotionChatbot
from medicine_scanning import scan_medicine
from transformers import pipeline as hf_pipeline
import subprocess
import torch
import platform
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# GPU
def get_available_gpus():
    """
    Get the GPU with the most available memory.
    Returns:
        int: The GPU index with the most available memory.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi error: {result.stderr}")

        free_memory = [int(x) for x in result.stdout.strip().split("\n")]
        
        # Find the GPU with the most free memory, if there are ties, choose the first one
        max_memory = max(free_memory)
        best_gpu = free_memory.index(max_memory)
        return best_gpu
    except Exception as e:
        print(f"Error selecting GPU: {e}")
        return None

# Select the GPU with the most available memory
best_gpu = get_available_gpus()
if best_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using GPU {best_gpu} with the most available memory.")
else:
    print("No GPU selected. Using default settings.")

# Check if CUDA is available and print the selected GPU
if torch.cuda.is_available():
    print(f"Selected GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU.")

# Hugging Face
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

print("Loading Mistral 7B starts...")
use_quantization = platform.system() != "Windows"

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)

if torch.cuda.is_available():
    print("CUDA available, loading model on GPU...")
    try:
        if use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )
            print("Model loaded with 4-bit quantization.")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )
            print("Model loaded in float16 on GPU.")
    except Exception as e:
        print(f"Error loading model on GPU: {e}. Falling back to CPU.")
        model = AutoModelForCausalLM.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
        print("Model loaded on CPU.")
else:
    print("CUDA not available, loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
    print("Model loaded on CPU.")

chatbot = EmotionChatbot(model, tokenizer)

# Use the Cerulean theme for vibrant blue accents
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CERULEAN],
    suppress_callback_exceptions=True,
    title="Medic AI"
)
server = app.server

# Navbar (white text on blue)
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Medic AI", className="fw-bold text-dark"),
        dbc.Nav([
            dbc.Button("Home", href="/", className="btn-custom me-2"),
            dbc.Button("About Us", href="/about", className="btn-custom me-2"),
            dbc.Button("Scan Medicine", href="/scan", className="btn-custom me-2"),
            dbc.Button("Chat with Therapist", href="/therapy", className="btn-custom")
        ], navbar=True)
    ]),
    color="primary",
    dark=True,
    className="mb-4"
)

# ---- Pages ----

home_page = html.Div(
    className="hero",
    children=html.Div(
        className="overlay",
        children=[
            html.H1("Welcome to Medic AI"),
            html.P("AI-powered medicine scanning and mental health support."),
            dbc.Button("Scan Medicine", href="/scan", className="btn-custom me-2"),
            dbc.Button("Chat with Therapist", href="/therapy", className="btn-custom")
        ]
    )
)


about_page = dbc.Container([
    html.H2("About Us", className="mb-3"),
    html.P("Semantic Bard Group — making healthcare accessible with AI."),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Img(src="/assets/team-zwe.jpg", className="team-img mb-2"),
                html.H5("Zwe Htet"),
                html.P("Team Member")
            ]), sm=4
        ),
        dbc.Col(
            html.Div([
                html.Img(src="/assets/team-mya.jpg", className="team-img mb-2"),
                html.H5("Mya Mjechal"),
                html.P("Team Member")
            ]), sm=4
        ),
        dbc.Col(
            html.Div([
                html.Img(src="/assets/team-htet.jpg", className="team-img mb-2"),
                html.H5("William"),
                html.P("Team Member")
            ]), sm=4
        ),
    ], className="g-4"),
    html.Hr(),
    html.P("Medical Advisor: Dr. Khin Lay Phyu, M.B.,B.S")
], className="page-content")

scan_page = dbc.Container([
    html.H2("AI Medicine Scanning", className="mb-3"),
    html.P("Snap or upload a photo of your medication label to get a summary."),

    # Chat‐style window
    html.Div(
        id="scan-window",
        className="chat-window",
        children=[
            html.Div(
                "Please upload a medicine image to begin.",
                className="bot-msg"
            )
        ]
    ),
    # Upload button (no text input)
    dcc.Upload(
        id='upload-image',
        children=html.Button("Click to upload image", id="upload-btn", className="send-btn"),
        accept="image/*",
        multiple=False,
        className="mt-2"
    ),

    # Add a hidden loading signal
    dcc.Store(id='image-uploaded', data=False),
    
    # Loading spinner
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=html.Div(id="scan-result")
    )
], className="page-content")

therapy_page = dbc.Container([
    html.H2("AI Therapist Chatbot", className="mb-3"),
    html.P("Private, empathetic chat—just type below."),
    html.Div(id="chat-window", className="chat-window", children=[
        html.Div("Hi there! How are you feeling today?", className="bot-msg")
    ]),
    dcc.Textarea(
        id='therapy-input',
        className="therapy-input",
        placeholder="Type your message here...",
    ),
    html.Button("Send", id='therapy-send', n_clicks=0, className="send-btn mt-2")
], className="page-content")

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content') 
])

# ---- Router callback ----
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/about':
        return about_page
    elif pathname == '/scan':
        return scan_page
    elif pathname == '/therapy':
        return therapy_page
    return home_page

# ---- Scan Upload / Display Uploaded Image / Scan Another ----

# Show Uploaded Image + Control Upload Button
@app.callback(
    [Output('scan-window', 'children'), 
     Output('image-uploaded', 'data'),
     Output('upload-btn', 'disabled')],
    Input('upload-image', 'contents'),
    State('scan-window', 'children')
)
def update_scan_window(uploaded_contents, current_children):
    current_children = current_children or []

    if not uploaded_contents:
        raise dash.exceptions.PreventUpdate

    current_children.append(
        html.Div(
            html.Img(
                src=uploaded_contents,
                style={'maxWidth': '200px', 'borderRadius': '8px'}
            ),
            className="user-msg",
            style={'clear': 'both', 'marginTop': '1rem'}
        )
    )

    # Disable Upload button after user uploads
    return current_children, uploaded_contents, True

# Perform Medicine Scan and Show Result
@app.callback(
    [Output('scan-result', 'children'),
     Output('upload-btn', 'disabled')],
    Input('image-uploaded', 'data')
)
def scan_and_generate(contents):
    if not contents:
        raise dash.exceptions.PreventUpdate

    drug_name, summary_text, error = scan_medicine(contents, model, tokenizer)

    if error:
        return html.Div(error, className="bot-msg"), False  # Enable button again if error

    return html.Div([
        html.Div([
            html.Strong(f"Summary for {drug_name}:"),
            html.P(summary_text)
        ], className="bot-msg", style={'marginBottom': '1rem', 'backgroundColor': '#eaf4fb', 'padding': '10px', 'borderRadius': '8px'})
    ]), False  # Enable Upload button again after finish

# ---- Therapy chat callback  ----
@app.callback(
    [Output('chat-window', 'children'), Output('therapy-input', 'value')],
    Input('therapy-send', 'n_clicks'),
    State('therapy-input', 'value'),
    State('chat-window', 'children')
)
def update_therapy(n, msg, children):
    if n and msg:
        children = children or []
        children.append(html.Div(msg, className="user-msg"))
        ai_response = asyncio.run(chatbot.chat(msg))
        children.append(html.Div(ai_response, className="bot-msg"))
        return children, ""
    return children, ""


if __name__ == '__main__':
    app.run(debug=True)
