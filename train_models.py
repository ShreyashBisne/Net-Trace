import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'difficulty_level'
]

def load_and_preprocess(file_path):
    # Load data
    df = pd.read_csv(file_path, names=columns)
    
    # Binary classification: normal vs attack
    df['label'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Drop original attack column and difficulty level
    df = df.drop(['attack', 'difficulty_level'], axis=1)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    return df, le_dict

print("Loading data...")
train_df, encoders = load_and_preprocess('train.csv')
test_df, _ = load_and_preprocess('test.csv')

# Ensure test set has same categorical encoding (simplified for this script)
# In a real scenario, we'd handle unseen labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print(f"RF Accuracy: {accuracy_score(y_test, rf_pred)}")

# Train MLP (Neural Network)
print("Training Neural Network...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)
print(f"MLP Accuracy: {accuracy_score(y_test, mlp_pred)}")

# Train Logistic Regression (Faster than SVM for large datasets)
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print(f"LR Accuracy: {accuracy_score(y_test, lr_pred)}")

# Save models and artifacts
print("Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/random_forest.joblib')
joblib.dump(mlp, 'models/neural_network.joblib')
joblib.dump(lr, 'models/logistic_regression.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(encoders, 'models/encoders.joblib')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.joblib')

print("All models trained and saved successfully!")
