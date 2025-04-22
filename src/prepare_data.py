#Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

#Function to laod data from Student_performance_data.csv
def load_data(file_path):

    #Loads the dataset
    df = pd.read_csv(file_path)
    
    #Remove Student not needed for analysis
    if 'StudentID' in df.columns:
        df.drop('StudentID', axis=1, inplace=True)
    
    #Checks for missing values
    nullvalues = df.isnull().sum()
    
    #Prints missing values per column
    print("Missing values per column:\n", nullvalues)
    
    #Return the cleaned dataframe and missing values
    return df, nullvalues

#Funtion to replace missing values with needed data
def replace_missing(df):

    #Split colunms into groups
    parental_columns = ['ParentalSupport', 'ParentalEducation']
    categorical_columns = ['Gender', 'Ethnicity', 'Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
    numerical_columns = ['StudyTimeWeekly', 'Absences', 'Age']

    #For loop to replace missing values with the mean
    for column in parental_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mean())
            print(f'Replaced missing values in {column}.')
        else:
            print(f'No missing values found in {column}.')

    #For loop to replace missing values with the mode
    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])
            print(f'Replaced missing values in {column}.')
        else:
            print(f'No missing values found in {column}.')

    #For loop to replace missing values with the mean
    for column in numerical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mean())
            print(f'Replaced missing values in {column}.')
        else:
            print(f'No missing values found in {column}.')
            
    return df

#Function that creates boxplots for visualization for outliers
def outlier_checking(df, numcols):
    outlier_plotting = df.melt(value_vars=numcols)
    sns.boxplot(data=outlier_plotting, x='value', y='variable', orient='h')
    plt.title('Box Plot: Numerical Features')
    plt.show()
    
#Function that removes outliers using (IQR Method)
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


#Function to show GPA performance based on various activities
def analyze_performance_and_visualize(data_path):
    #Loads the dataset
    df = pd.read_csv(data_path)

    #Lists activities to analyze
    acts = ["Extracurricular", "Sports", "Music", "Volunteering"]

    #Generate box plots for GPA distribution by activity
    for activity in acts:
        sns.boxplot(x=df[activity], y=df["GPA"])
        plt.title(f"GPA Distribution by {activity}")
        plt.xlabel(activity)
        plt.ylabel("GPA")
        plt.show()



#Function that preprocesses and saves data to various files
def preprocess_and_save_data(input_path, output_data_path, output_scaler_path):
    #Reads the dataset
    df = pd.read_csv(input_path)

    #Perform one-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=['Ethnicity', 'ParentalEducation', 'ParentalSupport'])

    #Convert boolean columns to integers
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    #Scales numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'StudyTimeWeekly', 'Absences']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Moves GradeClass to the end of the dataframe
    grade = df.pop('GradeClass')
    df['GradeClass'] = grade

    #Save the scaler to a file
    with open(output_scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    #Save the engineered dataset to a CSV file
    df.to_csv(output_data_path, index=False)



