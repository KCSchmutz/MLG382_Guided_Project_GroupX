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

    df2=df.drop(columns=["StudentID", "GradeClass"])
    # Add a mask to transform the categorical features
    # Replace values of categorical features that are not encoded
    df2['Gender'] = df2['Gender'].replace({0: 'Male', 1: 'Female'})
    df2['Ethnicity'] = df2['Ethnicity'].replace({0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'})
    df2['ParentalEducation'] = df2['ParentalEducation'].replace({0: 'None', 1: 'High School', 2: 'Some College', 3: 'Bachelors', 4: 'Higher Study'})
    df2['ParentalSupport'] = df2['ParentalSupport'].replace({0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'})
    
    # Feature Engineering: 
    # Add 3 new columns to the DataFrame
    df3 = df2
    # The 1st feature is to determine if the student outside activities are productive
    df3['Constructive_Extracurricular'] = df3.apply(lambda row: row['Extracurricular'] or ((row['Sports'] or row['Music']) or row['Volunteering']), axis=1)
    # The 2nd feature is to determine if a student is receiving support in their school life
    limiting_performance = ['None', 'Low', 'Moderate']
    df3['Receives_Support'] = df3.apply(lambda row:0 if ((row['Tutoring'] == 0) and (row['ParentalSupport'] in limiting_performance)) else 1, axis=1)
    # The 3rd is to determine the amount of study hours the student is busy catching up on work perweek
    weeks_in_a_year = 52
    df3['Catch_up_study_hours'] = df3.apply(lambda row: np.round(row['StudyTimeWeekly'] ** ((row['Absences'] / weeks_in_a_year)),2), axis=1)

    
    return df