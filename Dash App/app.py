import dash
from dash import html, dcc, Output, Input, State
import pickle
import numpy as np
import dash_bootstrap_components as dbc
import os

# Define the base path to the artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

# Load models and scaler using corrected paths
with open(os.path.join(ARTIFACTS_DIR, "regression_model.pkl"), "rb") as f:
    logistic_model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "deep_learning_model.pkl"), "rb") as f:
    deep_model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "random_forest_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "regression_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Theme colors
main_bg = "#242423"
card_bg = "#363635"
accent_color = "#FCDE9C"
button_color = "#FFA552"
text_color = "#BA5624"

# Input styles
input_style = {
    'marginBottom': '10px',
    'borderRadius': '8px',
    'border': 'none',
    'padding': '10px',
    'width': '100%',
    'color': 'black'
}

dropdown_style = {
    'marginBottom': '10px',
    'borderRadius': '8px',
    'border': 'none',
    'padding': '10px',
    'width': '100%',
    'backgroundColor': 'white',
    'color': 'black'
}

# Layout
app.layout = html.Div([
    html.H2("Student Grade Class Prediction", style={
        'textAlign': 'center',
        'color': accent_color,
        'marginBottom': '30px'
    }),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Age", style={'color': accent_color}),
                    dcc.Input(id='age', type='number', min=15, max=18, style=input_style),

                    html.Label("Gender", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='gender',
                        options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
                        style=dropdown_style
                    ),

                    html.Label("Study Time Weekly (hrs)", style={'color': accent_color}),
                    dcc.Input(id='study', type='number', min=0, max=168, style=input_style),

                    html.Label("Absences (0–30)", style={'color': accent_color}),
                    dcc.Input(id='absences', type='number', min=0, max=30, style=input_style),

                    html.Label("Ethnicity", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3}],
                        style=dropdown_style
                    ),

                    html.Label("Parental Education Level", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='parent_edu',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': 'Bachelors', 'value': 3},
                            {'label': 'Higher Study', 'value': 4}],
                        style=dropdown_style
                    ),

                    html.Label("Parental Support Level", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='parent_support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4}],
                        style=dropdown_style
                    ),

                    html.Label("Activities (check all that apply)", style={'color': accent_color}),
                    dcc.Checklist(
                        id='activities',
                        options=[
                            {'label': 'Tutoring', 'value': 'tutoring'},
                            {'label': 'Extracurricular', 'value': 'extracurricular'},
                            {'label': 'Sports', 'value': 'sports'},
                            {'label': 'Music', 'value': 'music'},
                            {'label': 'Volunteering', 'value': 'volunteering'},
                        ],
                        value=[],
                        style={'color': 'white', 'marginBottom': '20px'}
                    ),

                    html.Button("Predict", id='predict-btn', n_clicks=0, style={
                        'backgroundColor': button_color,
                        'border': 'none',
                        'borderRadius': '8px',
                        'padding': '10px 20px',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'width': '100%'
                    }),

                    html.Div(id='output', style={
                        'marginTop': '20px',
                        'fontSize': '20px',
                        'color': accent_color,
                        'textAlign': 'center'
                    })
                ], style={
                    'backgroundColor': card_bg,
                    'padding': '30px',
                    'borderRadius': '20px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.3)'
                })
            ], width=6)
        ], justify='center')
    ])
], style={'backgroundColor': main_bg, 'minHeight': '100vh', 'paddingTop': '50px'})


@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('study', 'value'),
    State('absences', 'value'),
    State('ethnicity', 'value'),
    State('parent_edu', 'value'),
    State('parent_support', 'value'),
    State('activities', 'value'),
)
def make_prediction(n_clicks, age, gender, study, absences, ethnicity, parent_edu, parent_support, activities):
    if n_clicks == 0:
        return ""

    # Validation
    if any(val is None for val in [age, gender, study, absences, ethnicity, parent_edu, parent_support]):
        return "⚠️ Please fill out all fields."

    if not (15 <= age <= 18):
        return "⚠️ Age must be between 15 and 18."
    if not (0 <= study <= 168):
        return "⚠️ Study time must be between 0 and 168 hours."
    if not (0 <= absences <= 30):
        return "⚠️ Absences must be between 0 and 30."

    # Activities
    features = {
        'tutoring': 1 if 'tutoring' in activities else 0,
        'extracurricular': 1 if 'extracurricular' in activities else 0,
        'sports': 1 if 'sports' in activities else 0,
        'music': 1 if 'music' in activities else 0,
        'volunteering': 1 if 'volunteering' in activities else 0
    }

    input_data = [
        age, gender, study, absences, ethnicity,
        parent_edu, parent_support,
        features['tutoring'], features['extracurricular'],
        features['sports'], features['music'], features['volunteering']
    ]

    # Scale input
    scaled = scaler.transform([input_data])

    # Predict from all models
    predictions = {
        "Logistic Regression": logistic_model.predict(scaled)[0],
        "Deep Learning": deep_model.predict(scaled)[0],
        "Random Forest": rf_model.predict(scaled)[0],
        "XGBoost": xgb_model.predict(scaled)[0],
    }

    return html.Div([
        html.Div(f"✅ {model_name}: {pred}", style={'marginBottom': '10px'})
        for model_name, pred in predictions.items()
    ])


if __name__ == '__main__':
    app.run(debug=True)
