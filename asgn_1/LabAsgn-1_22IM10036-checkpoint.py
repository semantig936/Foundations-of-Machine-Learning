# Name: Semanti Ghosh
# Roll No - 22IM10036
#Python Library Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#Load the Dataset
df = pd.read_csv('diabetes.csv')
df.head()

#Data Presentation
df.info()
df.describe(include = 'all').T
df.corr()

#Split the Data Frame into Features and Target
X = df.drop(columns=['Outcome']) # features
y = df['Outcome'] #Target
X.head()

#Applying StandardScalar
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
print(X.shape)
#Split Dataset into train and test (80%train, 20%test)
X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2)

#Implementation of Logistic Regression Classifier
diabetic_classifier_lr = LogisticRegression()
diabetic_classifier_lr.fit(X_tr, y_tr)

# Make predictions on the test dataset
y_pred = diabetic_classifier_lr.predict(X_te)

# Evaluate model performance
accuracy_dia = accuracy_score(y_te, y_pred)
precision_dia = precision_score(y_te, y_pred, zero_division=1)
recall_dia = recall_score(y_te, y_pred, zero_division=1)
f1_dia = f1_score(y_te, y_pred, zero_division=1)

# Create confusion matrix and classification report
conf_matrix_dia = confusion_matrix(y_te, y_pred)
class_report_dia = classification_report(y_te, y_pred)

# Results
print(f"Accuracy: {accuracy_dia}")
print(f"Precision: {precision_dia}")
print(f"Recall: {recall_dia}")
print(f"F1 Score: {f1_dia}")
print("Confusion Matrix:")
print(conf_matrix_dia)
print("Classification Report:")
print(class_report_dia)