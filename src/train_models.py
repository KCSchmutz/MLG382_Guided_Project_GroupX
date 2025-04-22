#Importing all required libraries
import importlib
import pickle 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor



# Function to split the dataset dropping column (Objective is to predict profit)
def split_features_target(df):
    target_column =['GradeClass_A', 'GradeClass_B','GradeClass_C','GradeClass_D','GradeClass_F']
    X = df.drop(target_column)
    Y = df[target_column]
    return X, Y
# Function to save the feature importance
def save_feature_importance(model, save_csv=True):
    features = model.named_steps["logisticregression"].get_feature_names_out()
    coefs = model.named_steps["logisticregression"].coef_[0]
    importance = pd.DataFrame({"feature": features, "importance": np.exp(coefs)})
    if save_csv:
        importance_path = "../data/feature_importance"
        if not os.path.exists(importance_path):
            importance.to_csv(importance_path, index=False)
        # Only saves if the files don't already exist
        
# Funtion to save feature names (Web App use)
def save_feature_list(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)


# Function to train split data
def create_train_test_split(X, Y, test_size=0.2, random_state=2000, save_csv=True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Boolean check to see if train.csv and test.csv already exists
    if save_csv:
        train_path = "../data/train.csv"
        test_path = "../data/test.csv"

        # Only saves if the files don't already exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            train_df = X_train.copy()
            train_df["GradeClass"] = Y_train

            test_df = X_test.copy()
            test_df["GradeClass"] = Y_test

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

    return X_train, X_test, Y_train, Y_test

# Function to reload the module and train the model
def get_model(model_type):

    # Reload the module to reflect changes (Don't have to restart Kernel everytime if this is included)
    import train_models
    importlib.reload(train_models)

    # Train the model based on the model_type specified
    if model_type == "logistic_regresion":

        #Calls the function to train a RandomForestRegressor model
        model=train_models.train_logistic_regresion()

    elif model_type == "random_forest":

        #Calls the function to train a RandomForestRegressor model
        model=train_models.train_random_forest()

    elif model_type == "adaboost":

        #Calls the function to train a RandomForestRegressor model
        model=train_models.train_adaboost()

    elif model_type == "xgboost":

        #Calls the function to train a RandomForestRegressor model
        model=train_models.train_xgboost()

    elif model_type == 'deep_learning':

        #Calls the function to train a deep learning model
        model=train_models.train_deep_learning_model()
    return model


#Function to create a preprocessor for categorical features 
def get_preprocessor(X):

    #Identify categorical columns (columns with datatype of 'object' and 'category')
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    #Creates a preprocessing pipeline for categorical columns with OneHotEncoder (Changes categorical data into numerical data)
    preprocessor = ColumnTransformer(
        transformers=[ 
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        
        #Leave numerical columns unchanged in the output
        remainder='passthrough'  
    )
    return preprocessor


#Function that trains, evaluates, saves and visualizes models
def train_model(model, model_name, X_train, X_test, Y_train, Y_test, output_dir="../artifacts"):

    #Creating preprocessor for categorical features
    preprocessor = get_preprocessor(X_train)

    if (model_name == 'deep_learning'):
        #Adding preprocessing model into one pipeline
        pipeline = make_pipeline(preprocessor, model)
    else:
        #Only encoding the pyplines catagorical features
        pipeline = make_pipeline(model)

    #Trains entire pipeline
    pipeline.fit(X_train, Y_train)

    #Predicts test set results
    Y_pred = pipeline.predict(X_test)

    #Prints evaluatoin metrics for each model
    print(f"Model: {model_name}")

    #Used to measure how well predictions match actual values
    print(f"R2 Score: {r2_score(Y_test, Y_pred):.4f}") 

    #Prints average magnitude of prediction error
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.2f}")

    #Prints large errors
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred):.2f}")
    
    #Prints the accoracy score for the model
    model_train_acc = pipeline.score(X_train, Y_train)
    model_test_acc = pipeline.score(X_test, Y_test)

    print(f"{model_name} : Train accuracy Score: { model_train_acc:.4f}")
    print(f"{model_name} : Test accuracy Score: { model_test_acc:.4f}")
    #Prints spacing between metrics and graphs
    print("-" * 50)

    #Save a model to a specific pkl file
    filepath = os.path.join(output_dir, f"{model_name}_model.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to: {filepath}")

    # Save predictions
    results_df = pd.DataFrame({
        "Model": model_name,
        "Actual": Y_test.values,
        "Predicted": Y_pred
    })

    predictions_path = f"../artifacts/{model_name}_predictions.csv"
    if os.path.exists(predictions_path):
        results_df.to_csv(predictions_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(predictions_path, index=False)

    #Scatter plot (Shows Actual Values against Predicted Values)
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Actual GradeClass")
    plt.ylabel("Predicted GradeClass")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    #Line plot (Shows Actual Values against Predicted Values)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(Y_test)), Y_test, label="Actual", marker='o')
    plt.plot(range(len(Y_pred)), Y_pred, label="Predicted", marker='x')
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.xlabel("Count")
    plt.ylabel("GradeClass")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Function that trains RandomForestRegressor model
def train_random_forest():
    model = make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42))
    return model
    

#Function that trains GradientBoostingRegressor model
def train_logistic_regresion():
    model = LogisticRegression(solver='liblinear', max_iter=100)
    return model

#Function that trains AdaBoostRegressor model
def train_adaboost():
    # Using DecisionTreeRegressor as the base estimator
    base_estimator = DecisionTreeRegressor(max_depth=3)
    model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, learning_rate=0.1, random_state=42)
    return model

#Function that trains XGBRegressor model
def train_xgboost():
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    return model

#Function that trains Deep learning model
def train_deep_learning_model():
    model = MLPRegressor(hidden_layer_sizes=(5, 4), activation='relu', solver='sgd', max_iter=500, random_state=43)
    return model
