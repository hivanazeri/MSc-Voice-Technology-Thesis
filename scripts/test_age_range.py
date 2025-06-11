import pandas as pd
import numpy as np
import os
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Function to load test data from a CSV file into a pandas DataFrame
def load_data(csv_path):
    print(f"Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path)  # Read CSV file into DataFrame
    return df

# Function to preprocess test data
# Drops the 'filename' column since it is not used for prediction
def preprocess_data(df):
    print("Preprocessing test data...")
    df = df.drop(columns=['filename'])  # Remove unnecessary column
    return df

# Function to run predictions on the test set and evaluate the model
# Prints accuracy and classification report, saves reports and confusion matrix plot
def evaluate(model, label_encoder, X_test, y_test, output_dir):
    print("Running predictions and evaluating...")

    # Predict class labels for the test features
    y_pred = model.predict(X_test)

    # Convert encoded numeric labels back to original label strings for interpretability
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # Calculate accuracy score of predictions
    acc = accuracy_score(y_test_labels, y_pred_labels)

    # Generate detailed classification report (precision, recall, f1-score for each class)
    report = classification_report(y_test_labels, y_pred_labels)

    # Print accuracy and classification report to console
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # Save the classification report to a text file in the output directory
    with open(os.path.join(output_dir, "test_classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Compute confusion matrix to visualize prediction errors and successes
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    # Plot confusion matrix as a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test_labels),
                yticklabels=np.unique(y_test_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()

    # Save the plot image to the output directory
    plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"))
    plt.close()  # Close plot to free memory

# Main function to orchestrate test data loading, preprocessing, model loading, and evaluation
def main(args):
    # Load test dataset from CSV file path
    df = load_data(args.csv_path)

    # Preprocess test data by dropping unnecessary columns
    df_processed = preprocess_data(df)

    # Load the saved trained Random Forest model from disk
    model = joblib.load(args.model_path)

    # Load the saved label encoder for the target variable (age) from disk
    label_encoder = joblib.load(args.encoder_path)

    # Encode the test target labels using the loaded label encoder
    y_test = label_encoder.transform(df_processed['age'])

    # Select features by dropping the target column 'age'
    X_test = df_processed.drop(columns=['age'])

    # Run model evaluation on the test dataset
    evaluate(model, label_encoder, X_test, y_test, args.output_dir)

# If this script is run directly, parse command-line arguments and execute main()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Random Forest model for age prediction")

    # Required argument: path to CSV file containing test data features and labels
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with test features")

    # Required argument: path to saved Random Forest model file (.joblib)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved Random Forest model (.joblib)")

    # Required argument: path to saved label encoder file (.joblib)
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to the saved label encoder (.joblib)")

    # Optional argument: directory where evaluation outputs will be saved (default "results")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation outputs")

    args = parser.parse_args()  # Parse command-line arguments

    # Ensure output directory exists or create it
    os.makedirs(args.output_dir, exist_ok=True)

    # Run main function with parsed arguments
    main(args)
