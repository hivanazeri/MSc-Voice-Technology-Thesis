# Age Range Prediction from Voice

This project predicts a speaker's age range based on their voice using machine learning techniques. It extracts acoustic features from human voice recordings and trains a Random Forest classifier to categorize age groups. By combining signal processing and machine learning, the project aims to identify vocal markers of aging.

## Project Structure

```
age-range-voice-prediction/
├── data/                # CSV datasets
├── models/              # Trained model & label encoder
├── scripts/             # Scripts for training, testing, and feature extraction
├── test-results/        # Evaluation results on test set
├── train-output/        # Output plots and metrics from training
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/age-range-voice-prediction.git
cd age-range-voice-prediction
```

### 2. Install dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r scripts/requirements.txt
```

## Running the Model

### Extract features 
'''bash
python scripts/extract_voice_features.py --input_dir data/clips --output_csv data/features.csv
'''

### Train the Model

```bash
python scripts/train_age_range_model.py
```

This will:
- Extract features
- Train a Random Forest model
- Save the trained model and label encoder in `models/`
- Generate evaluation plots in `train-output/`

### Test the Model

```bash
python scripts/test_age_range.py
```

This will:
- Load the trained model
- Run predictions on the test set
- Output metrics and confusion matrix in `test-results/`

## Results

- Accuracy: See `train-output/classification_report.txt`
- Confusion Matrix:  
  `train-output/confusion_matrix.png`
- Feature Importances:  
  `train-output/feature_importances.png`

## Features Used

- MFCCs, Zero-Crossing Rate, Spectral Bandwidth, RMS Energy
  
- Fundamental frequency (F0), Formants (F1, F2)

- Jitter, Shimmer, Spectral Tilt, Speech Rate

Feature extraction is handled by `scripts/extract_voice_features.py`.

## License

This project is licensed under the MIT License.

## Contributions

Pull requests are welcome. If you'd like to improve this repo or extend it, feel free to contribute.

## Contact

For questions or collaboration, please reach out via GitHub Issues.
