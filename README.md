# Fraud Detection using Synthetic Data

## Overview
This project aims to develop a machine learning model capable of detecting fraudulent transactions using synthetic data.

## Data Exploration
### Distribution of Transaction Amounts
![Transaction Amount Distribution](visualizations/transaction_amount_distribution.png)
This plot shows the distribution of transaction amounts. Most transactions fall within a specific range, with a few high-value transactions.

### Count of Fraudulent vs Non-Fraudulent Transactions
![Fraudulent vs Non-Fraudulent Transactions](visualizations/fraudulent_vs_non_fraudulent.png)
This plot highlights the imbalance between fraudulent and non-fraudulent transactions, with fraudulent transactions being a small minority.

## Data Preprocessing
### Class Distribution Before and After Oversampling
![Class Distribution Before and After Oversampling](visualizations/class_distribution_before_after_oversampling.png)
These plots show the class distribution before and after oversampling. Oversampling helps balance the classes, which is crucial for training an effective model.

## Model Evaluation
### Model Performance Metrics
```plaintext
              precision    recall  f1-score   support

       False       1.00      0.98      0.99       189
        True       0.98      1.00      0.99       191

    accuracy                           0.99       380
   macro avg       0.99      0.99      0.99       380
weighted avg       0.99      0.99      0.99       380
```
The classification report provides precision, recall, and F1-score for both classes, showing that the model performs well on both non-fraudulent and fraudulent transactions.

### ROC Curve
![ROC Curve](visualizations/roc_curve.png)
The ROC curve shows the trade-off between true positive rate and false positive rate for different threshold values. The area under the curve (AUC) is a measure of the model's ability to distinguish between classes, with a score of 1.0 indicating perfect performance.

## Results
- Model achieved a ROC-AUC score of 1.0 after handling class imbalance.
- Visualizations and detailed analysis can be found in the Jupyter Notebook.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud_detection_project.git
   ```

2. Navigate to the project directory and install dependencies:
   ```bash
   cd fraud_detection_project
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook to see the full analysis:
   ```bash
   jupyter notebook notebooks/fraud_detection_notebook.ipynb
   ```

## Author
Robert Grantham
