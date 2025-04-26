# app.py
import asyncio
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# IMPORT YOUR CHATBOT
from emotion_chat import EmotionChatbot

# Instantiate a single bot for the life of this app
chatbot = EmotionChatbot()

def chat_sync(message: str) -> str:
    """Run the async EmotionChatbot.chat in a blocking way for Dash callbacks."""
    return asyncio.run(chatbot.chat(message))

# --- Dash app setup ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CERULEAN],
    suppress_callback_exceptions=True,
    title="Medic AI"
)
server = app.server

# --- Navbar ---
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

# --- Pages ---
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
    # ... rest unchanged ...
], className="page-content")

scan_page = dbc.Container([
    html.H2("AI Medicine Scanning", className="mb-3"),
    html.P("Snap or upload a photo of your medication label to get a summary."),
    html.Div(id="scan-window", className="chat-window", children=[
        html.Div("Please upload a medicine image to begin.", className="bot-msg")
    ]),
    dcc.Upload(
        id='upload-image',
        children=html.Button("Click to upload image", className="send-btn"),
        accept="image/*",
        multiple=False,
        className="mt-2"
    )
], className="page-content")

therapy_page = dbc.Container([
    html.H2("AI Therapist Chatbot", className="mb-3"),
    html.P("Private, empathetic chat—just type below."),
    # Loading spinner around the chat window
    dcc.Loading(
        id="therapy-loading",
        type="default",
        children=html.Div(
            id="chat-window",
            className="chat-window",
            children=[html.Div("Hi there! How are you feeling today?", className="bot-msg")]
        )
    ),
    # Store to hold session history
    dcc.Store(id="therapy-history", data=[]),
    # Input area
    dcc.Textarea(
        id='therapy-input',
        className="therapy-input",
        placeholder="Type your message here...",
    ),
    html.Button("Send", id='therapy-send', n_clicks=0, className="send-btn mt-2")
], className="page-content")

# --- App layout & routing ---
app.layout = html.Div([dcc.Location(id='url', refresh=False), navbar, html.Div(id='page-content')])

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

# --- Scan upload callback (unchanged) ---
@app.callback(
    Output('scan-window', 'children'),
    Input('upload-image', 'contents'),
    State('scan-window', 'children')
)
def update_scan(contents, children):
    if not contents:
        return children
    children = children or []
    # user bubble
    children.append(html.Div(html.Img(src=contents, style={'maxWidth':'200px','borderRadius':'8px'}),
                             className="user-msg", style={'clear':'both','marginTop':'1rem'}))
    # fake summary
    children.append(html.Div([
        html.Strong("Summary for Xanax (Alprazolam):"),
        html.Ul([
            html.Li("Use: Anxiety relief"),
            html.Li("Dosage: 0.25–0.5 mg orally, up to three times daily"),
            html.Li("Warnings: Risk of dependence; avoid alcohol")
        ])
    ], className="bot-msg", style={'clear':'both','marginTop':'0.5rem'}))
    return children

# --- Therapy chat callback (integrated) ---
@app.callback(
    Output('chat-window', 'children'),
    Output('therapy-history', 'data'),
    Output('therapy-input', 'value'),
    Input('therapy-send', 'n_clicks'),
    State('therapy-input', 'value'),
    State('therapy-history', 'data')
)
def update_therapy(n_clicks, msg, history):
    if not n_clicks or not msg:
        # no change
        return dash.no_update, history, dash.no_update

    # initialize history list if needed
    history = history or []

    # append user's message
    history.append({"user": msg})

    # get AI response
    try:
        bot_resp = chat_sync(msg)
    except Exception as e:
        bot_resp = "Sorry, something went wrong. Please try again."

    # attach bot response
    history[-1]["bot"] = bot_resp

    # rebuild chat-window
    children = []
    for turn in history:
        children.append(html.Div(turn["user"], className="user-msg"))
        children.append(html.Div(turn["bot"], className="bot-msg"))

    # return updated window, history store, and clear input
    return children, history, ""

if __name__ == '__main__':
    app.run(debug=True)
