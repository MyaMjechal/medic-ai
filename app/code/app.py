import dash
import asyncio
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from emotion_chat_2 import EmotionChatbot
from medicine_scanning import scan_medicine


chatbot = EmotionChatbot()

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

# ---- Scan upload callback ----
# Show Uploaded Image Immediately
@app.callback(
    [Output('scan-window', 'children'), Output('image-uploaded', 'data')],
    Input('upload-image', 'contents'),
    State('scan-window', 'children')
)
def display_uploaded_image(contents, children):
    if not contents:
        return children, False
    children = children or []
    children.append(
        html.Div(
            html.Img(
                src=contents,
                style={'maxWidth': '200px', 'borderRadius': '8px'}
            ), 
                className="user-msg",
                style={'clear': 'both', 'marginTop': '1rem'}
            )
        )
    return children, contents

# Process Medicine Scan
@app.callback(
    Output('scan-result', 'children'),
    Input('image-uploaded', 'data')
)
def scan_and_generate(contents):
    if not contents:
        return ""
    drug_name, summary_text, error = scan_medicine(contents)
    if error:
        return html.Div(error, className="bot-msg")
    return html.Div([
        html.Strong(f"Summary for {drug_name}:"),
        html.P(summary_text)
    ], className="bot-msg")

# Disable Upload Button During Scan
@app.callback(
    Output('upload-btn', 'disabled'),
    Input('image-uploaded', 'data')
)
def disable_upload_button(data):
    return bool(data)

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
