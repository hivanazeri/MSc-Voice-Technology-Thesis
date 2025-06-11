import pandas as pd
import numpy as np
import os
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
# Function to load data from a CSV file into a pandas DataFrame
def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df
# Function to preprocess the DataFrame:
# - drops unnecessary columns
# - encodes categorical variables (gender and age) into numeric labels
# - separates features and target label
def preprocess_data(df):
    print("Preprocessing data...")
    # Drop filename
    df = df.drop(columns=['filename'])

    # Encode categorical features
    le_gender = LabelEncoder()
    le_age = LabelEncoder()
    # Convert gender from strings to numeric labels (e.g., male->0, female->1)
    df['gender'] = le_gender.fit_transform(df['gender'])
    # Convert age classes from strings to numeric labels
    df['age'] = le_age.fit_transform(df['age'])

    # Separate features and labels
    # Features: all columns except 'age' (target)
    X = df.drop(columns=['age'])
    y = df['age']

    return X, y, le_age
# Function to train a Random Forest classifier on the training data
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf
# Function to evaluate the trained model on the test data
def evaluate_model(model, X_test, y_test, label_encoder):
    print("Evaluating model...")
    y_pred = model.predict(X_test) # Predict labels for test set
    # Convert numeric labels back to original string labels for better readability in report
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Calculate accuracy score of predictions
    acc = accuracy_score(y_test_labels, y_pred_labels)
    # Generate a detailed classification report (precision, recall, F1-score)
    report = classification_report(y_test_labels, y_pred_labels)

    print(f"\nAccuracy: {acc:.4f}")
    print("Classification Report:\n", report)
# Function to save the trained model and the label encoder to disk for later use
def save_model(model, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "random_forest_model.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")
 # Save model and encoder using joblib
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")

# Main function to orchestrate data loading, preprocessing, training, evaluation, and saving
def main(args):
    # Preprocess dataset
    df = load_data(args.csv_path)
    X, y, label_encoder = preprocess_data(df)

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model on training data
    model = train_model(X_train, y_train)
    # Evaluate model performance on test data
    evaluate_model(model, X_test, y_test, label_encoder)

    # Save model and encoder
    save_model(model, label_encoder, args.output_dir)
# Entry point of the script when run from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model for age prediction from voice features")
    # Command-line argument for the CSV path containing features and labels
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with features")
    # Command-line argument for output directory to save model and encoder (default 'models')
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the trained model and encoder")

    args = parser.parse_args() # Parse command-line arguments
    main(args) # Run main with the parsed arguments




# To get Feature_importances.png

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def plot_selected_feature_importances(model, feature_names, output_dir):
#     # Define the features you want to plot
#     selected_features = ['formant2', 'formant1', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
#                          'speech rate', 'f0 mean', 'spectral tilt', 'shimmer', 'jitter']
    
#     # Get all feature importances from the model
#     importances = model.feature_importances_
    
#     # Create a dictionary mapping feature_name -> importance
#     feature_importance_dict = dict(zip(feature_names, importances))
    
#     # Filter only the selected features (and handle if any not found)
#     filtered_features = []
#     filtered_importances = []
#     for feat in selected_features:
#         if feat in feature_importance_dict:
#             filtered_features.append(feat)
#             filtered_importances.append(feature_importance_dict[feat])
#         else:
#             print(f"Warning: Feature '{feat}' not found in model features.")
    
#     # Sort selected features by importance descending
#     sorted_idx = np.argsort(filtered_importances)[::-1]
#     sorted_features = [filtered_features[i] for i in sorted_idx]
#     sorted_importances = [filtered_importances[i] for i in sorted_idx]
    
#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.title("Selected Feature Importances")
#     plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
#     plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
#     plt.tight_layout()
    
#     # Save the figure
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, "selected_feature_importances.png"))
#     plt.close()

