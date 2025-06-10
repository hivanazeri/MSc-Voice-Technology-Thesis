import pandas as pd
import numpy as np
import os
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop filename
    df = df.drop(columns=['filename'])

    # Encode categorical features
    le_gender = LabelEncoder()
    le_age = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['age'] = le_age.fit_transform(df['age'])

    # Separate features and labels
    X = df.drop(columns=['age'])
    y = df['age']

    return X, y, le_age

def train_model(X_train, y_train):
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, label_encoder):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels)

    print(f"\nAccuracy: {acc:.4f}")
    print("Classification Report:\n", report)

def save_model(model, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "random_forest_model.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")


def main(args):
    df = load_data(args.csv_path)
    X, y, label_encoder = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)

    # Save model and encoder
    save_model(model, label_encoder, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model for age prediction from voice features")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with features")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the trained model and encoder")

    args = parser.parse_args()
    main(args)
