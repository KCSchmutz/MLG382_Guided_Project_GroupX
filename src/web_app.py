import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, '..', 'src')

def get_DLmodel():
    return load_model(os.path.join(artifacts_dir, 'deep_learning_model.h5'))

def get_randomforest_model():
    with open(os.path.join(artifacts_dir, 'random_forest_model.pkl'), 'rb') as f:
        return pickle.load(f)

def get_regression_model():
    with open(os.path.join(artifacts_dir, 'logistic_regression_model.pkl'), 'rb') as f:
        return pickle.load(f)

def get_xgboost_model():
    with open(os.path.join(artifacts_dir, 'xgboost_model.pkl'), 'rb') as f:
        return pickle.load(f)

with open(os.path.join(artifacts_dir, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

def get_scaler():
    with open(os.path.join(artifacts_dir, 'scaler_test.pkl'), 'rb') as f:
        return pickle.load(f)

# Theme colors
main_bg = "#242423"
card_bg = "#363635"
accent_color = "#FCDE9C"
button_color = "#FFA552"
text_color = "#BA5624"

# Styles
input_style = {
    'marginBottom': '10px',
    'borderRadius': '8px',
    'border': 'none',
    'padding': '10px',
    'width': '100%',
    'color': 'black'
}
dropdown_style = input_style.copy()
dropdown_style['backgroundColor'] = 'white'

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H2("Student Grade Class Prediction", style={
        'textAlign': 'center', 'color': accent_color, 'marginBottom': '30px'
    }),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Age", style={'color': accent_color}),
                    dcc.Input(id='age', type='number', style=input_style),

                    html.Label("Gender", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='gender',
                        options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
                        style=dropdown_style
                    ),

                    html.Label("Study Time Weekly (hrs)", style={'color': accent_color}),
                    dcc.Input(id='study_time', type='number', style=input_style),

                    html.Label("Absences", style={'color': accent_color}),
                    dcc.Input(id='absences', type='number', style=input_style),

                    html.Label("Ethnicity", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3}
                        ],
                        style=dropdown_style
                    ),

                    html.Label("Parental Education Level", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='parental_education',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': 'Bachelors', 'value': 3},
                            {'label': 'Higher Study', 'value': 4}
                        ],
                        style=dropdown_style
                    ),

                    html.Label("Parental Support Level", style={'color': accent_color}),
                    dcc.Dropdown(
                        id='parental_support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4}
                        ],
                        style=dropdown_style
                    ),

                    html.Label("Activities", style={'color': accent_color}),
                    dcc.Checklist(
                        id='activities',
                        options=[
                            {'label': 'Tutoring', 'value': 'tutoring'},
                            {'label': 'Extracurricular', 'value': 'extracurricular'},
                            {'label': 'Sports', 'value': 'sports'},
                            {'label': 'Music', 'value': 'music'},
                            {'label': 'Volunteering', 'value': 'volunteering'}
                        ],
                        value=[],
                        style={'color': 'white', 'marginBottom': '20px'}
                    ),

                    html.Button("Predict", id='predict_button', n_clicks=0, style={
                        'backgroundColor': button_color,
                        'border': 'none',
                        'borderRadius': '8px',
                        'padding': '10px 20px',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'width': '100%'
                    }),

                    html.Div(id='prediction-output', style={
                        'marginTop': '20px',
                        'fontSize': '18px',
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
    Output('prediction-output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('study_time', 'value'),
    State('absences', 'value'),
    State('ethnicity', 'value'),
    State('parental_education', 'value'),
    State('parental_support', 'value'),
    State('activities', 'value')
)
def predict_grade(n_clicks, age, gender, study_time, absences, ethnicity, parental_education, parental_support, activities):
    if n_clicks == 0:
        return ""

    if None in (age, gender, study_time, absences, ethnicity, parental_education, parental_support):
        return "⚠️ Please fill in all required fields."

    activity_flags = {
        'Tutoring': 1 if 'tutoring' in activities else 0,
        'Extracurricular': 1 if 'extracurricular' in activities else 0,
        'Sports': 1 if 'sports' in activities else 0,
        'Music': 1 if 'music' in activities else 0,
        'Volunteering': 1 if 'volunteering' in activities else 0
    }

    input_data = {
        'Age': [age],
        'Gender': [gender],
        'StudyTimeWeekly': [study_time],
        'Absences': [absences],
        'Tutoring': [activity_flags['Tutoring']],
        'Extracurricular': [activity_flags['Extracurricular']],
        'Sports': [activity_flags['Sports']],
        'Music': [activity_flags['Music']],
        'Volunteering': [activity_flags['Volunteering']],
        **{f'Ethnicity_{i}': [1 if ethnicity == i else 0] for i in range(4)},
        **{f'ParentalEducation_{i}': [1 if parental_education == i else 0] for i in range(5)},
        **{f'ParentalSupport_{i}': [1 if parental_support == i else 0] for i in range(5)}
    }

    input_df = pd.DataFrame(input_data)
    scaler = get_scaler()
    input_df[['Age', 'StudyTimeWeekly', 'Absences']] = scaler.transform(input_df[['Age', 'StudyTimeWeekly', 'Absences']])

    dl_model = get_DLmodel()
    rf_model = get_randomforest_model()
    logistic_model = get_regression_model()
    xgb_model = get_xgboost_model()

    log_prediction = logistic_model.predict(input_df)
    rf_prediction = rf_model.predict(input_df)
    xgb_prediction = xgb_model.predict(input_df)
    dl_prediction = dl_model.predict(input_df)

    if len(dl_prediction.shape) == 2 and dl_prediction.shape[1] > 1:
        class_prediction = np.argmax(dl_prediction)
        probability = np.max(dl_prediction)
    else:
        class_prediction = int(round(float(dl_prediction[0][0])))
        probability = float(dl_prediction[0][0]) if class_prediction == 1 else 1 - float(dl_prediction[0][0])

    probability_percent = probability * 100

    return html.Div([
        html.Div(f"✅ Logistic Regression: {log_prediction[0]}", style={'marginBottom': '10px'}),
        html.Div(f"✅ Random Forest: {rf_prediction[0]}", style={'marginBottom': '10px'}),
        html.Div(f"✅ XGBoost: {xgb_prediction[0]}", style={'marginBottom': '10px'}),
        html.Div(f"✅ Deep Learning: {class_prediction} (Confidence: {probability_percent:.2f}%)")
    ])

if __name__ == "__main__":
    app.run(debug=True)
