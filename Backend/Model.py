import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('survey.csv')  # Apne file path daal

# Select 5 columns
columns = ['Age', 'Gender', 'treatment', 'work_interfere', 'mental_health_consequence']
df = df[columns]

# Basic info
print(df.head())
print(df.info())
print(df.isnull().sum())

# Age: Remove outliers
df = df[(df['Age'] >= 18) & (df['Age'] <= 80)]

# Gender: Standardize
df['Gender'] = df['Gender'].str.lower().replace({
    'male': 'Male', 'm': 'Male', 'female': 'Female', 'f': 'Female',
    'woman': 'Female', 'man': 'Male'
}, regex=True)
# Non-standard genders ko 'Other' mein convert karo
df['Gender'] = df['Gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')

# work_interfere: Handle missing values
df['work_interfere'].fillna('Unknown', inplace=True)

# Check cleaned data
print(df.isnull().sum())
print(df['Gender'].value_counts())
print(df['work_interfere'].value_counts())

from sklearn.preprocessing import LabelEncoder

# Encode Gender
le_gender = LabelEncoder()
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])

# Encode treatment
df['treatment_encoded'] = df['treatment'].map({'No': 0, 'Yes': 1})

# Encode work_interfere
df['work_interfere_encoded'] = df['work_interfere'].map({
    'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Unknown': -1
})

# Encode mental_health_consequence (target)
le_target = LabelEncoder()
df['mental_health_consequence_encoded'] = le_target.fit_transform(df['mental_health_consequence'])

joblib.dump(le_target, 'le_target.pkl')

# Select features and target
features = ['Age', 'Gender_encoded', 'treatment_encoded', 'work_interfere_encoded']
X = df[features]
y = df['mental_health_consequence_encoded']

# Check encoded data
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model

joblib.dump(model, 'mental_health_model.pkl')
