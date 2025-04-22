#Importing Required Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


#Function to split features and target columns
def split_features_target(input_path, test_size=0.2, random_state=42):
    #Loads the dataset
    df = pd.read_csv(input_path)
    
    #Separate target and features
    y = df['GradeClass']
    features = df.drop(columns=['GradeClass'])
    
    #Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=test_size, random_state=random_state, stratify=y)

    
    return X_train, X_test, y_train, y_test



def logistic_regression_model(X_train, y_train, X_test, y_test):
    #Initialize the model
    LogReg = LogisticRegression(max_iter=20000, random_state=42)
    
    #Train the model
    LogReg.fit(X_train, y_train)
    
    #Make predictions
    y_pred_LogReg = LogReg.predict(X_test)
    
    #Print classification report
    print("Logistic Regression:\n", classification_report(y_test, y_pred_LogReg))
    
    #Save the trained model
    with open("../artifacts/LogisticRegression_model.pkl", "wb") as file:
        pickle.dump(LogReg, file)
    print("Logistic Regression model saved to ../artifacts/LogisticRegression_model.pkl")



def random_forest_model(X_train, y_train, X_test, y_test):
    #Initialize and train the Random Forest model
    RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
    RandomForest.fit(X_train, y_train)

    #Predict and generate the classification report
    y_pred_RandomForest = RandomForest.predict(X_test)
    print("Random Forest:\n", classification_report(y_test, y_pred_RandomForest))

    #Save the trained model
    with open("../artifacts/RandomForestRegressor_model.pkl", "wb") as file:
        pickle.dump(RandomForest, file)
    print(f"Random Forest model saved to ../artifacts/RandomForestRegressor_model.pkl")




def xgboost_model(X_train, y_train, X_test, y_test):
    #Initialize and train the XGBoost model
    XGBoost = XGBClassifier(eval_metric='mlogloss')
    XGBoost.fit(X_train, y_train)

    #Predict and generate the classification report
    y_pred_XGBoost = XGBoost.predict(X_test)
    print("XGBoost:\n", classification_report(y_test, y_pred_XGBoost))

    #Save the trained model
    with open("../artifacts/XGBoostRegressor_model.pkl", "wb") as file:
        pickle.dump(XGBoost, file)
    print(f"XGBoost model saved to ../artifacts/XGBoostRegressor_model.pkl")




def deep_learning_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    #Convert labels to categorical format for deep learning
    y_train_DeepLearning = to_categorical(y_train)
    y_test_DeepLearning = to_categorical(y_test)
    
    #Build the deep learning model
    model = Sequential()
    
    #First Dense layer will handle input shape automatically
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Assuming 5 classes

    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train the model
    model.fit(X_train, y_train_DeepLearning, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_DeepLearning))

    #Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_DeepLearning)
    print("Test Accuracy:", accuracy)

    #Make predictions
    y_pred_DeepLearning = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_DeepLearning, axis=1)

    #Print classification report and confusion matrix
    print(classification_report(y_test, y_pred_classes))
    print(confusion_matrix(y_test, y_pred_classes))

    #Save the trained model
    model.save("../artifacts/DeepLearning_model_.h5")
    print(f"Deep Learning model saved to ../artifacts/DeepLearning_model_.h5")


