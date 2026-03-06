# 📧 Email Spam Detection

Machine learning project for detecting **spam and legitimate emails** using NLP and Logistic Regression.

The project includes training on a labeled email dataset and testing the trained model on **real Gmail emails through the Gmail API**.

> Gmail integration is read-only. No emails are deleted, moved, or modified.

## 🚀 Features

- Email parsing and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Word-level TF-IDF
- Character-level TF-IDF
- Logistic Regression
- Cross-validation
- Threshold analysis
- Gmail API integration
- Real-world email prediction

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Gmail API

## 📊 Results

Final model:

| Metric | Score |
|---|---:|
| Accuracy | **97.24%** |
| ROC-AUC | **99.72%** |
| Spam Precision | **97%** |
| Spam Recall | **95%** |
| Spam F1 | **96%** |

Using **Word + Character TF-IDF** improved performance compared with Word TF-IDF alone.

## 📬 Gmail Integration

The trained model can read recent Gmail messages and produce predictions:

```text
Sender              Subject                    Probability   Prediction
myABL               Fund Transfer Alert             1.23%   HAM
LinkedIn            Sr. AI Engineer...              2.81%   HAM
Unknown             You won $1,000,000!            98.71%   SPAM
```

The Gmail integration is intentionally **read-only**.

## 📂 Structure

```text
EmailSpamDetection/
├── data/
├── models/
├── notebooks/
└── src/
    ├── parse_email.py
    ├── predict_email.py
    ├── gmail_reader.py
    └── predict_gmail.py
```

## ▶️ Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train/prepare the dataset:

```bash
python src/parse_email.py
```

Test an email:

```bash
python src/predict_email.py
```

Test real Gmail emails:

```bash
python src/predict_gmail.py
```

## 👤 Author

**Muizz Karim**