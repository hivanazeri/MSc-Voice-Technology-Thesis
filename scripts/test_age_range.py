import pandas as pd
import numpy as np
import os
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(csv_path):
    print(f"Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    print("Preprocessing test data...")
    df = df.drop(columns=['filename'])

    return df

def evaluate(model, label_encoder, X_test, y_test, output_dir):
    print("Running predictions and evaluating...")
    y_pred = model.predict(X_test)

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # Save classification report
    with open(os.path.join(output_dir, "test_classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test_labels),
                yticklabels=np.unique(y_test_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"))
    plt.close()

def main(args):
    # Load test data
    df = load_data(args.csv_path)
    df_processed = preprocess_data(df)

    # Load label encoder and model
    model = joblib.load(args.model_path)
    label_encoder = joblib.load(args.encoder_path)

    # Encode target variable
    y_test = label_encoder.transform(df_processed['age'])
    X_test = df_processed.drop(columns=['age'])

    evaluate(model, label_encoder, X_test, y_test, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Random Forest model for age prediction")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with test features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved Random Forest model (.joblib)")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to the saved label encoder (.joblib)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation outputs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
