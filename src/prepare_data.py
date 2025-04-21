#Importing all required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Function to load data
def load_data(filepath):
    #Loads data into the DataFrame
    df = pd.read_csv(filepath)
    
    #Drop rows and columns that are completely null
    df = df.dropna(how='all')  # Drop rows that are all NaN
    df = df.dropna(axis=1, how='all')  # Drop columns that are all NaN

    #Fill missing numeric values with mean, categorical with mode
    for column in df.columns:

        #Boolean to calculate the sum of all null columns found
        if df[column].isnull().sum() > 0:

            #Boolean to check if column datatype is float64 or int64
            if df[column].dtype in ['float64', 'int64']:

                #Fills columns with mean
                df[column] = df[column].fillna(df[column].mean())
            else:

                #Fills columns with mode
                df[column] = df[column].fillna(df[column].mode()[0])

    # Drops duplicate rows
    df = df.drop_duplicates()

    df=df.drop(columns=["StudentID", "GradeClass"])
    # Add a mask to transform the categorical features
    # Replace values of categorical features that are not encoded
    df['Gender'] = df['Gender'].replace({0: 'Male', 1: 'Female'})
    df['Ethnicity'] = df['Ethnicity'].replace({0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'})
    df['ParentalEducation'] = df['ParentalEducation'].replace({0: 'None', 1: 'High School', 2: 'Some College', 3: 'Bachelors', 4: 'Higher Study'})
    df['ParentalSupport'] = df['ParentalSupport'].replace({0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'})
    
    return df.head(10)

def catagorical_column_transformations(df):
    # Add a mask to transform the categorical features
    # Replace values of categorical features that are not encoded
    df['Tutoring'] = df['Tutoring'].replace({0: 'No', 1: 'Yes'})
    df['Extracurricular'] = df['Extracurricular'].replace({0: 'No', 1: 'Yes'})
    df['Sports'] = df['Sports'].replace({0: 'No', 1: 'Yes'})
    df['Music'] = df['Music'].replace({0: 'No', 1: 'Yes'})
    df['Volunteering'] = df['Volunteering'].replace({0: 'No', 1: 'Yes'})
    return df

def feature_engineering(df):
    # Feature Engineering: 
    # Add 3 new columns to the DataFrame
    # The 1st feature is to determine if the student outside activities are productive
    df['Constructive_Extracurricular'] = df.apply(lambda row:'Yes' if ((row['Extracurricular']=='Yes') or (((row['Sports']=='Yes') or (row['Music']=='Yes')) or (row['Volunteering']=='Yes'))) else 'No', axis=1)
    # The 2nd feature is to determine if a student is receiving support in their school life
    limiting_performance = ['None', 'Low', 'Moderate']
    df['Receives_Support'] = df.apply(lambda row:'No' if ((row['Tutoring'] == 'No') and (row['ParentalSupport'] in limiting_performance)) else 'Yes', axis=1)
    # The 3rd is to determine the amount of study hours the student is busy catching up on work perweek
    weeks_in_a_year = 52
    df['Catch_up_study_hours'] = df.apply(lambda row: np.round(row['StudyTimeWeekly'] ** ((row['Absences'] / weeks_in_a_year)),2), axis=1)
    return df    