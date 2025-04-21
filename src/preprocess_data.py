#Importing all required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Function to scale features using StandardScaler and Encoded using OneHotEncoder ~ Normalising the data
def scale_and_encode(df):
    #Initialization of StandardScaler
    scaler = StandardScaler()
    numeric_features = df.columns.drop(["GPA" ,"Age", "Gender", "Ethnicity", "ParentalEducation", "ParentalSupport", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering", "Constructive_Extracurricular", "Receives_Support"])
    # update the cols with their normalized values
    scaler.fit(df[numeric_features])
    df[numeric_features] = scaler.transform(df[numeric_features])
    #Identify categorical columns (columns with datatype of 'object' and 'category')
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_transformed = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_transformed

#Fuction that removes anomalies using Isolation Forest
def remove_anomalies(df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    numeric_features = df.columns.drop(["GPA" ,"Age", "Gender", "Ethnicity", "ParentalEducation", "ParentalSupport", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering", "Constructive_Extracurricular", "Receives_Support"])
    outliers = iso.fit_predict(df[numeric_features])
    df_cleaned = df[outliers == 1]
    return df_cleaned

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
    output_var = 'GPA'
    numeric_data = df.select_dtypes(include=['number']).columns.tolist()
    numeric_features = [col for col in numeric_data if col != output_var]
    return numeric_features

def make_Onehot(df):
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        df[col] = df[col].replace({'False': 0.0, 'True': 1.0})
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

