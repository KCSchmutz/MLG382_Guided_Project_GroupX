#Importing all required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from category_encoders import OneHotEncoder


#Function to scale features using StandardScaler and Encoded using OneHotEncoder ~ Normalising the data
def scale_and_encode(df):
    
    ohe = OneHotEncoder(
    use_cat_names=True, 
    cols=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'ParentalSupport']
    )
    # Transform data
    df_transformed = ohe.fit_transform(df)

    #Initialization of StandardScaler
    scaler = StandardScaler()
    scalar_features = df.columns.drop(["Age", "Gender", "Ethnicity", "ParentalEducation", "ParentalSupport", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering", "Constructive_Extracurricular", "Receives_Support"])
    # update the cols with their normalized values
    scaler.fit(df_transformed[scalar_features])
    df_transformed[scalar_features] = scaler.transform(df_transformed[scalar_features])
    return df_transformed

#Fuction that removes anomalies using Isolation Forest
def remove_anomalies(df):
    iso = IsolationForest(contamination=0.01, random_state=42)
    outliers = iso.fit_predict(df)
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