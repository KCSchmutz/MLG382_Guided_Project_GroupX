import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Initialize model and scaler manually (mock model for demo purposes only)
model = XGBClassifier()
scaler = StandardScaler()

# Grade class labels
grade_labels = ['A', 'B', 'C', 'D', 'F']

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("BrightPath Student Grade Predictor", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Age (15–18)"), dcc.Input(id='age', type='number', min=15, max=18), html.Br(),
        html.Label("Gender (0 = Male, 1 = Female)"), dcc.Input(id='gender', type='number', min=0, max=1), html.Br(),
        html.Label("Ethnicity (0=Caucasian, 1=African American, 2=Asian, 3=Other)"), dcc.Input(id='ethnicity', type='number', min=0, max=3), html.Br(),
        html.Label("Parental Education (0–4)"), dcc.Input(id='pedu', type='number', min=0, max=4), html.Br(),
        html.Label("Weekly Study Time (0–20 hours)"), dcc.Input(id='study', type='number', min=0, max=20), html.Br(),
        html.Label("Absences (0–30)"), dcc.Input(id='absences', type='number', min=0, max=30), html.Br(),
        html.Label("Tutoring (0 = No, 1 = Yes)"), dcc.Input(id='tutoring', type='number', min=0, max=1), html.Br(),
        html.Label("Parental Support (0–4)"), dcc.Input(id='psupport', type='number', min=0, max=4), html.Br(),
        html.Label("Extracurricular (0 = No, 1 = Yes)"), dcc.Input(id='extra', type='number', min=0, max=1), html.Br(),
        html.Label("Sports (0 = No, 1 = Yes)"), dcc.Input(id='sports', type='number', min=0, max=1), html.Br(),
        html.Label("Music (0 = No, 1 = Yes)"), dcc.Input(id='music', type='number', min=0, max=1), html.Br(),
        html.Label("Volunteering (0 = No, 1 = Yes)"), dcc.Input(id='vol', type='number', min=0, max=1), html.Br(),
        html.Button("Predict Grade", id='submit', n_clicks=0)
    ], style={'columnCount': 2, 'margin': '20px'}),

    html.Div(id='output', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '30px'})
])

@app.callback(
    Output('output', 'children'),
    Input('submit', 'n_clicks'),
    State('age', 'value'), State('gender', 'value'), State('ethnicity', 'value'),
    State('pedu', 'value'), State('study', 'value'), State('absences', 'value'),
    State('tutoring', 'value'), State('psupport', 'value'),
    State('extra', 'value'), State('sports', 'value'),
    State('music', 'value'), State('vol', 'value')
)
def predict_grade(n_clicks, age, gender, ethnicity, pedu, study, absences,
                tutoring, psupport, extra, sports, music, vol):

    features = [age, gender, ethnicity, pedu, study, absences,
                tutoring, psupport, extra, sports, music, vol]

    if None in features:
        return "Please fill in all fields to get a prediction."

    input_data = np.array([features])
    input_scaled = scaler.fit_transform(input_data)  # Fit just to avoid error (replace with actual trained scaler)
    prediction = model.fit(input_scaled, [0]).predict(input_scaled)[0]  # Dummy fit + predict for placeholder
    probability = 1.0  # Placeholder

    return f"Predicted Grade: {grade_labels[prediction]} (Confidence: {probability:.2%})"

if __name__ == '__main__':
    app.run(debug=True)
