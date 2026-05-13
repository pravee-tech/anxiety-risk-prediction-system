from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Flask App Initialization
app = Flask(__name__)
warnings.filterwarnings("ignore")

# Load Dataset
try:
    df = pd.read_csv("enhanced_anxiety_dataset.csv")
except FileNotFoundError:
    print(" ERROR: 'enhanced_anxiety_dataset.csv' not found!")
    exit()

# Keep only useful features
df = df.drop(columns=['id'], errors='ignore')
df = df.dropna(subset=['Anxiety Level (1-10)'])

# Store categorical options for form
original_cols = df.columns.drop(['Anxiety Level (1-10)'], errors='ignore')
categorical_cols_map = {
    'Gender': df['Gender'].dropna().unique().tolist(),
    'Occupation': df['Occupation'].dropna().unique().tolist(),
    'Smoking': ['Yes', 'No'],
    'Family History of Anxiety': ['Yes', 'No'],
    'Dizziness': ['Yes', 'No'],
    'Medication': ['Yes', 'No'],
    'Recent Major Life Event': ['Yes', 'No']
}

# Preprocessing
encoders = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Fill missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Risk Mapping
def map_risk(x):
    if x <= 4: return 0  # Low
    elif x <= 7: return 1  # Moderate
    else: return 2  # High

df['risk_alert'] = df['Anxiety Level (1-10)'].apply(map_risk)

# Features & Labels
X = df.drop(columns=['Anxiety Level (1-10)', 'risk_alert'], errors='ignore')
y = df['risk_alert']
training_columns = X.columns

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Model Training
# LightGBM with tuning
lgbm = LGBMClassifier(
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    class_weight='balanced',
    random_state=42
)
lgbm.fit(X_train, y_train)

# RandomForest with tuning
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate models
rf_pred = rf_model.predict(X_test)
lgbm_pred = lgbm.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
lgbm_acc = accuracy_score(y_test, lgbm_pred)

print("\n Model Training Complete")
print(f" RandomForest Accuracy: {rf_acc:.4f}")
print(f" LightGBM Accuracy: {lgbm_acc:.4f}")
print("\n RandomForest Report:\n", classification_report(y_test, rf_pred))
print("\n LightGBM Report:\n", classification_report(y_test, lgbm_pred))


# Choose best model
best_model = lgbm if lgbm_acc >= rf_acc else rf_model
print(f"\n Using {'LightGBM' if best_model == lgbm else 'RandomForest'} for predictions.")

# Suggestions Logic
def get_personalized_suggestions(user_input_df):
    data_point = user_input_df.iloc[0].copy()

    # Apply encoders
    for col, le in encoders.items():
        if col in data_point:
            if data_point[col] not in le.classes_:
                data_point[col] = le.classes_[0]
            data_point[col] = le.transform([data_point[col]])[0]

    # Convert numerics
    for col in training_columns:
        if col not in categorical_cols_map:
            data_point[col] = pd.to_numeric(data_point[col], errors='coerce')

    # Ensure correct columns
    data_point_processed = pd.DataFrame([data_point], columns=training_columns).fillna(0)

    # Scale
    data_point_scaled = scaler.transform(data_point_processed)

    # Predict
    prediction_value = int(best_model.predict(data_point_scaled)[0])
    risk_mapping = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
    risk_label = risk_mapping[prediction_value]

    # Suggestions
    suggestions = []
    if prediction_value == 2:
        suggestions.append({'text': 'Connect with a therapist', 'icon': 'fa-solid fa-comments'})
        suggestions.append({'text': 'Practice urgent stress-relief techniques', 'icon': 'fa-solid fa-wind'})
    elif prediction_value == 1:
        suggestions.append({'text': 'Deep breathing exercises', 'icon': 'fa-solid fa-lungs'})
        suggestions.append({'text': 'Mindfulness exercises', 'icon': 'fa-solid fa-brain'})

    if data_point_processed['Stress Level (1-10)'].iloc[0] > 7:
        suggestions.append({'text': 'Your stress is very high. Try meditation daily.', 'icon': 'fa-solid fa-person-praying'})
    if data_point_processed['Sleep Hours'].iloc[0] < 6:
        suggestions.append({'text': 'Improve your sleep hygiene.', 'icon': 'fa-solid fa-bed'})
    if data_point_processed['Physical Activity (hrs/week)'].iloc[0] < 2.5:
        suggestions.append({'text': 'Increase physical activity for better mood.', 'icon': 'fa-solid fa-person-running'})

    if not suggestions:
        suggestions.append({'text': 'Maintain a balanced lifestyle.', 'icon': 'fa-solid fa-scale-balanced'})
        suggestions.append({'text': 'Monitor your stress regularly.', 'icon': 'fa-solid fa-chart-line'})

    return risk_label, prediction_value, suggestions

# Flask Routes
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/test", methods=["GET", "POST"])
def index():
    prediction_data = {'risk_label': None, 'prediction_value': None, 'suggestions': None}
    user_inputs = {}

    if request.method == "POST":
        user_inputs = request.form.to_dict()
        user_df = pd.DataFrame([user_inputs])

        risk_label, prediction_value, suggestions = get_personalized_suggestions(user_df)
        prediction_data.update({
            'risk_label': risk_label,
            'prediction_value': prediction_value,
            'suggestions': suggestions
        })

    return render_template(
        "index.html",
        categorical_cols=categorical_cols_map,
        columns=original_cols,
        prediction_data=prediction_data,
        user_inputs=user_inputs
    )

# Run App
if __name__ == "__main__":
    app.run(debug=True)