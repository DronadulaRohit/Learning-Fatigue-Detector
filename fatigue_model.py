import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_fatigue_model(input_file='clustered_sessions.csv'):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run clustering.py first.")
        return

    # Features for prediction (Can we use focus_score? If we want to PREDICT risk before it happens or without subjective measure, maybe not.
    # But usually 'focus time' is given as input. Let's include everything except the label.)
    X = df[['session_duration_min', 'break_frequency', 'focus_score', 'time_of_day', 'time_since_last_break']]
    y = df['fatigue_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = rf.feature_importances_
    feature_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_imp)
    
    # Visualize Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='magma')
    plt.title('Feature Importance for Fatigue Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved to feature_importance.png")

if __name__ == "__main__":
    train_fatigue_model()
