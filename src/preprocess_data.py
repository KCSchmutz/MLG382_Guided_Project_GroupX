#Importing all required libraries
import pandas as pd
import random
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder

#Function to scale features using StandardScaler and Encoded using OneHotEncoder ~ Normalising the data
def scale_and_encode(df):
    # Reload the module to reflect changes (Don't have to restart Kernel everytime if this is included)
    scaler = StandardScaler()
    numeric_data = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [col for col in numeric_data]
    # update the cols with their normalized values
    scaler.fit(df[numeric_features])
    df[numeric_features] = scaler.transform(df[numeric_features])
    #Identify categorical columns (columns with datatype of 'object' and 'category')
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'GradeClass']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    cat=df.columns.drop(["GPA" ,"StudyTimeWeekly", "Absences", "Catch_up_study_hours"])
    return df

#Fuction that removes anomalies using Isolation Forest
def remove_anomalies(df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    numeric_data = df.select_dtypes(include=['number']).columns.tolist()
    #numeric_features = df.columns.drop(["Age", "Gender", "Ethnicity", "ParentalEducation", "ParentalSupport", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering", "Constructive_Extracurricular", "Receives_Support"])
    outliers = iso.fit_predict(df[numeric_data])
    df_cleaned = df[outliers == 1]
    return df_cleaned

# Function that encodes the Y column
def encode_target(df, target_column):
    encoded =[]
    temp=0.0
    for i in range(5):
        temp += i*0.2 + random.uniform(0.01, 0.001)
        encoded.append(temp)
    df['GradeClass'] = df.apply(lambda row: encoded[0] if (row[target_column]=='A') else encoded[1] if (row[target_column]=='B') else encoded[2] if (row[target_column]=='C') else encoded[3] if (row[target_column]=='D') else encoded[4], axis=1)
    return df
#Function to remove outliers
def remove_outliers(df, column):
    #Using Interquartile Range (IQR) Approach
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    #Acquiring minimim, maximimum ranges for each column
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #Returns the DataFrame to keep only the rows values in the IQR range
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_numeric_columns(df):
    numeric_data = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [col for col in numeric_data]
    return numeric_features

def make_Onehot(df):
    categorical_features=df.columns.drop(["GPA" ,"StudyTimeWeekly", "Absences", "Catch_up_study_hours"]).tolist()
    for col in categorical_features:
        df[col] = df[col].replace({False: 0.0, True: 1.0})
    return df

#Function that loops over and over to remove all outliers (Not all outliers are removed after one iteration)
def iterative_outlier_removal(df, numerical_columns):
    #While loop that continues looping to remove all outliers
    while True:

        #Variable prev_shape to holp the dataframe's shape
        prev_shape = df.shape

        #For loop to iterate over each numerical column
        for col in numerical_columns:

            #This runs the remove outliers function above
            df = remove_outliers(df, col)

        #Boolean check to see if the new shape is the same as the previous shape
        if df.shape == prev_shape:

            #If it is the loop breaks signaling that there are no more outliers
            break

    return df 
