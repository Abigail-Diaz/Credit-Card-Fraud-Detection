# Credit Card Fraud Detection

Team Members: Joshua Hanscom, Andrew Rivera and Abigail Diaz

Course: CS 4662 - Advanced Machine Learning and Deep Learning

Instructor: Professor Mohammad Pourhomayoun

Date: March 2026

## 1. Introduction

Credit card fraud is a significant problem in the financial sector, costing billions of dollars annually. Detecting fraudulent transactions in real-time is challenging due to the massive volume of legitimate transactions compared to a very small number of fraudulent ones. In this project, we analyze a dataset of European credit card transactions to build and evaluate models capable of identifying fraud.

Using this dataset, we pursue three main objectives:

- Classification: Apply machine learning models to predict whether a specific transaction is fraudulent or legitimate.

- Handling Extreme Imbalance: Implement data sampling techniques to address the severe class imbalance inherent in fraud detection data.

- Performance vs. Computation: Evaluate the trade-offs between model accuracy and the computational complexity required to train them on large datasets.

By comparing multiple modeling strategies, we aim to determine which methods deliver the most reliable detection of fraud without flagging too many legitimate transactions as false positives.

## 2. Dataset Description

The dataset, sourced from Kaggle (ULB Machine Learning Group), contains credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions over a two-day period.

The dataset is highly imbalanced, with only 492 fraudulent transactions, accounting for just 0.172% of the total data. Due to confidentiality reasons, the original features have been transformed using Principal Component Analysis (PCA).

Features: 28 numerical features (V1 through V28) which are the principal components obtained with PCA.

Un-transformed Features: Time (seconds elapsed between each transaction and the first transaction) and Amount (the transaction value).

Target Variable: Class, a binary variable indicating whether the transaction is fraudulent (1 = yes, 0 = no).

## 3. Data Preprocessing

To prepare the dataset for modeling, we will implement the following pipeline:

- Scaling: While the PCA features (V1-V28) are already scaled, the Time and Amount features are not. We will apply a robust scaling method to the Amount feature to mitigate the influence of extreme transaction outliers.

- Handling Class Imbalance: Because fraudulent transactions are extremely rare, training a model on the raw data will lead to bias. We will utilize sampling techniques (such as random undersampling of the majority class) to create a balanced subset of data for initial model training.

- Splitting the Dataset: We will perform a train-test split using a stratified approach to ensure the original ratio of fraud-to-normal transactions is preserved across both the training and testing sets.

## 4. Methods / Models Used

To thoroughly investigate this dataset, we will implement a tiered modeling strategy, starting with baseline classical algorithms and progressing to advanced ensemble and deep learning techniques.

A primary focus across all models will be addressing the extreme class imbalance. Alongside data sampling, we will heavily utilize model weight tuning (e.g., assigning higher class weights/penalties to fraudulent misclassifications) to force the algorithms to prioritize the minority class without needing to alter the underlying data distribution.

### Classical Baselines

- K-Nearest Neighbors (KNN): We will use KNN to establish a fundamental distance-based baseline for performance, allowing us to gauge the baseline complexity of separating the classes.

- Linear Support Vector Machine (SVM): Because the dataset features (V1-V28) have already undergone Principal Component Analysis (PCA), we will test a Linear SVM to determine if the PCA projection created a linearly separable space for fraudulent and normal transactions.

### Non-Linear & Ensemble Models

- Non-Linear SVM: If the classes are not linearly separable, we will apply a Non-Linear SVM using the kernel trick (e.g., Radial Basis Function) to capture complex decision boundaries.

- Random Forest & Gradient Boosted Trees: We will utilize powerful tree-based ensembles. Random Forest (bagging) will help reduce variance, while Gradient Boosting (boosting) will actively correct misclassifications. These models are highly resilient to imbalanced data and will serve as our primary classical classifiers.

### Deep Learning

- Artificial Neural Networks (ANN): To capture the most complex, underlying representations within the PCA features, we will design and train a multi-layer Artificial Neural Network. The network will be trained using backpropagation, with a focus on optimizing hidden layer architectures and activation functions. To prevent the network from simply predicting the majority "normal" class, we will implement custom loss functions and utilize weighted classes during the training phase.

### Cross-Validation & Hyperparameter Tuning Strategy

To ensure our models generalize to unseen data rather than memorizing the training set, rigorous cross-validation is mandatory across all tiers.

- Stratified K-Fold Cross-Validation: Because the dataset contains only 0.172% fraudulent transactions, a standard random split would likely result in validation folds with zero fraud cases. We will strictly use Stratified K-Fold to guarantee that the exact ratio of fraud-to-normal transactions is preserved perfectly across every training and validation split.

- Efficient Tuning: For computationally heavy models like the Non-Linear SVM and the ANN, we will pair our cross-validation with randomized search techniques on our sampled subsets to find optimal parameters without exceeding processing limits.

## 5. Results and Evaluation

Because 99.8% of the data belongs to the "Not Fraud" class, standard accuracy is not a valid metric for this project. Instead, our evaluation will rely on:

- AUPRC (Area Under the Precision-Recall Curve): The primary metric for highly imbalanced data.

- Recall (Sensitivity): To measure our ability to catch as many actual frauds as possible.

- F1-Score: To find the balance between precision and recall.

(results are pending...)

## 6. Discussion

Once models have been trained and evaluated, this section will discuss:

- Which models performed best at detecting fraud and minimizing false positives.

- The impact of data scaling and undersampling on model performance.

- The computational challenges faced when running distance-based algorithms on large datasets.

- Insights gained from the LLM-generated fraud explanations.

## 7. Conclusion

(To be completed: Summarize findings and possible future improvements for a production environment.)

### References

Credit Card Fraud Detection Dataset, Kaggle (MLG-ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Abigail-Diaz/Credit-Card-Fraud-Detection.git <local-name>
cd <local-name>
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

### 3. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

---

### 4. Add Virtual Environment to `.gitignore`

Create or update `.gitignore`:

```bash
.venv/
```

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6. Install Jupyter Notebook

```bash
pip install jupyter
```

---

### 7. Run Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebook file in your browser.

---

### Notes

- Make sure Python 3.8+ is installed
- Always activate the virtual environment before running the project
