
import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter
import joblib

#relevant ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#ML models
from sklearn.linear_model import LogisticRegression


#warning hadle
warnings.filterwarnings("ignore")
data = pd.read_csv('loan_data.csv')
data.drop('Loan_ID',axis=1,inplace=True)
data.isnull().sum().sort_values(ascending=False)

#filling the missing data

null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


for col in null_cols:
    # print(f"{col}:\n{data[col].value_counts()}\n","-"*50)
    data[col] = data[col].fillna(
    data[col].dropna().mode().values[0] )   

    
data.isnull().sum().sort_values(ascending=False)

#list of all the columns.columns
#Cols = tr_df.tolist()
#list of all the numeric columns
num = data.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = data.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  data[num]
#categoric df
loan_cat = data[cat]



#converting categorical values to numbers

to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

# adding the new numeric values from the to_numeric variable to both datasets
tr_df = data.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

# convertind the Dependents column
Dependents_ = pd.to_numeric(tr_df.Dependents)


# dropping the previous Dependents column
tr_df.drop(['Dependents'], axis = 1, inplace = True)

# concatination of the new Dependents column with both datasets
tr_df = pd.concat([tr_df, Dependents_], axis = 1)




y = tr_df['Loan_Status']
X = tr_df.drop('Loan_Status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'path_to_your_model.pkl')

# Make predictions on new data
# Load the data
# data = pd.DataFrame({
#     'Gender': [1],
#     'Married': [2],
#     'Education': [1],
#     'Self_Employed': [2],
#     'ApplicantIncome': [19],
#     'CoapplicantIncome': [0.0],
#     'LoanAmount': [120.0],
#     'Loan_Amount_Term': [360.0],
#     'Credit_History': [1],
#     'Property_Area': [3],
#     'Dependents': [0]
# })

# prediction = model.predict(data)

# print(prediction)

# # Convert predictions to human-readable labels
# loan_status_labels = {
#     0: 'Not Approved',
#     1: 'Approved'
# }
# predicted_loan_status = loan_status_labels[prediction[0]]

# # Display the predicted loan status
# print("Predicted Loan Status:", predicted_loan_status)
