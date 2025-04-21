import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#Step 1: Import CSVs
df = pd.read_csv('data\Student_performance_data.csv', delimiter=",")

#Step 2: Separate features and target
X = df.drop(columns=["StudentID", "GradeClass", "GPA"])
y = df['GradeClass']

#Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=df['GradeClass'])

#Step 4: Build a pipeline of training
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder

#Step 4: Train XGBoost Model
pipeline = Pipeline(steps=[
    ('clf', XGBClassifier(eval_metric='mlogloss', random_state=8))
])
#pipe

#Step 5: Set up hyperparameter tuning
from skopt import BayesSearchCV
from skopt.space import Real,Categorical, Integer

search_space = {
    'clf__max_depth' : Integer(2,8),
    'clf__learning_rate' : Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample' : Real(0.5, 1.0),
    'clf__colsample_bytree' : Real(0.5, 1.0),
    'clf__colsample_bylevel' : Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha' : Real(0.0, 10.0),
    'clf__reg_lambda' : Real(0.0, 10.0),
    'clf__gamma' : Real(0.0, 10.0)
}

#Step 6: Training the XGBoost model
opt = BayesSearchCV(pipeline, search_space, cv=3, n_iter=10, scoring='accuracy', random_state=8)
#Can change cv and n_iter to higher values

opt.fit(X_train, y_train)

#Step 7: Evaluate and make predictions

opt.best_estimator_
opt.best_score_
opt.score(X_test, y_test)
opt.predict(X_test)
opt.predict_proba(X_test)

predictions = opt.predict(X_test)

#Grade mapping
grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
y_test_letters = pd.Series(y_test).map(grade_map)
y_pred_letters = pd.Series(predictions).map(grade_map)
label_names = [grade_map[i] for i in sorted(grade_map)]

#Step 8: Evaluation
accuracy = accuracy_score(y_test_letters, y_pred_letters)
print("XGBoost Accuracy:", round(accuracy, 4))
print("Classification Report:")
print(classification_report(y_test_letters, y_pred_letters, target_names=label_names, zero_division=0))

#Step 9: Evaulate feature importance
opt.best_estimator_.steps

from xgboost import plot_importance

xgboost_model = opt.best_estimator_.named_steps['clf']
plot_importance(xgboost_model)
plt.show()

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
labels = sorted(y.unique())  # Ensures label order matches

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

#Step 11: Save the model as pkl file in artifacts

with open("artifacts/xgboost_model.pkl", "wb") as file:
    pickle.dump(opt.best_estimator_, file)

#Step 12: Save predictions to CSV
comp_df = X_test.copy()
comp_df["Actual_GradeClass"] = y_test.values
comp_df["Predicted_GradeClass"] = predictions

#Step 13: Show and save the predictions table
try:
    from IPython.display import display

    # Prepare DataFrame for display
    comp_df = pd.DataFrame({"Actual": y_test.values,"Predicted": predictions})
    comp_df["Match"] = comp_df["Actual"] == comp_df["Predicted"]

    def highlight_false_text(row):
        styles = []
        for col in row.index:
            if col == "Match" and row["Match"] == False:
                styles.append("color: red; background-color: black")
            else:
                styles.append("background-color: black; color: white")
        return styles

    print("First 20 Predictions:")
    display(comp_df.head(20).style.apply(highlight_false_text, axis=1))

except Exception as e:
    print("\n First 20 Predictions:")
    print(comp_df.head(20).to_string(index=False))

comp_df.to_csv("/artifacts/xgboost_prediction.csv", index=False)