import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to pandas DataFrame
file_path = "D:/College Work/AI/SONAR Rock VS Mine/Sonar data.csv"
sonar_data = pd.read_csv(file_path, header=None)

# Display the first few rows of the dataframe
print(sonar_data.head())

# Number of rows and columns
print(sonar_data.shape)

# Descriptive statistics of the dataset
print(sonar_data.describe())

# Distribution of the labels
print(sonar_data[60].value_counts())

# Mean value for all the columns grouped by the label
print(sonar_data.groupby(60).mean())

# Separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

# Creating and training the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data:', test_data_accuracy)

# Example input data for prediction
input_data = (0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115)

# Changing the data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prediction using the trained model
prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'M':
    print('The object is Mine!!!! Be Careful')
else:
    print('The object is Rock')