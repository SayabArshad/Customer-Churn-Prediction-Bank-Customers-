import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Data
df = pd.read_csv("d:/python_ka_chilla/Internship/task 3/Churn_Modelling.csv")
print(df.head())

# Select features and target
X = df[['Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary']]
y = df['Exited']

# Encode categorical data
X_encoded = X.copy()
X_encoded = pd.get_dummies(X_encoded, columns=['Geography', 'Gender'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = X_encoded.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()
