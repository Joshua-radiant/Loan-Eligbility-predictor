import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("loan_eligibility_dataset.csv")

# ----------------------------
# Encode categorical columns
# ----------------------------
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Education'] = le.fit_transform(df['Education'])
df['Employment'] = le.fit_transform(df['Employment'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# ----------------------------
# Split features and target
# ----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# ----------------------------
# USER INPUT SECTION
# ----------------------------
print("\n=== LOAN ELIGIBILITY PREDICTOR ===")

age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
education = input("Enter Education (Graduate/Not Graduate): ")
employment = input("Enter Employment (Salaried/Self-Employed): ")
income = int(input("Enter Monthly Income: "))
loan_amount = int(input("Enter Loan Amount: "))
credit_score = int(input("Enter Credit Score: "))

# Encode user input
gender = 1 if gender.lower() == "male" else 0
education = 1 if education.lower() == "graduate" else 0
employment = 1 if employment.lower() == "self-employed" else 0

# Create input array
user_data = np.array([[age, gender, education, employment, income, loan_amount, credit_score]])
user_data_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_data_scaled)

# Output
if prediction[0] == 1:
    print("\n✅ Loan Approved")
else:
    print("\n❌ Loan Rejected")
