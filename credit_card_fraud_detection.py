{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "62147c44",
   "metadata": {},
   "source": [
    "# Credit Card Fraud Detection\n",
    "Team Members: Joshua Hanscom,  Andrew Rivera and Abigail Diaz\n",
    "\n",
    "Course: CS 4662 - Advanced Machine Learning and Deep Learning\n",
    "\n",
    "Instructor: Professor Mohammad Pourhomayoun\n",
    "\n",
    "Date: March 2026"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3def1049",
   "metadata": {},
   "source": [
    "## 1. Introduction\n",
    "Credit card fraud is a significant problem in the financial sector, costing billions of dollars annually. Detecting fraudulent transactions in real-time is challenging due to the massive volume of legitimate transactions compared to a very small number of fraudulent ones. In this project, we analyze a dataset of European credit card transactions to build and evaluate models capable of identifying fraud.\n",
    "\n",
    "Using this dataset, we pursue three main objectives:\n",
    "\n",
    "- Classification: Apply machine learning models to predict whether a specific transaction is fraudulent or legitimate.\n",
    "\n",
    "- Handling Extreme Imbalance: Implement data sampling techniques to address the severe class imbalance inherent in fraud detection data.\n",
    "\n",
    "- Performance vs. Computation: Evaluate the trade-offs between model accuracy and the computational complexity required to train them on large datasets.\n",
    "\n",
    "By comparing multiple modeling strategies, we aim to determine which methods deliver the most reliable detection of fraud without flagging too many legitimate transactions as false positives."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55edb52a-e669-4ac5-89fc-4c59de3ae76c",
   "metadata": {},
   "source": [
    "# 1.1 Configuration\n",
    "\n",
    "We define global configuration values used throughout the project, such as random seeds and common parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "46a18f40-c01b-423f-83da-6233131b76b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "TEST_SIZE = 0.2\n",
    "NUM_FOLDS = 10\n",
    "\n",
    "# for reproducibility\n",
    "\n",
    "np.random.seed(RANDOM_STATE)\n",
    "\n",
    "def save_results(model_name, scores, file_path=\"results.csv\"):\n",
    "    results = pd.DataFrame({\n",
    "        \"model\": [model_name],\n",
    "        \"mean_f1\": [scores[\"test_f1\"].mean()],\n",
    "        \"std_f1\": [scores[\"test_f1\"].std()],\n",
    "        \"mean_recall\": [scores[\"test_recall\"].mean()],\n",
    "        \"std_recall\": [scores[\"test_recall\"].std()],\n",
    "        \"mean_auprc\": [scores[\"test_auprc\"].mean()],\n",
    "        \"std_auprc\": [scores[\"test_auprc\"].std()],\n",
    "    })\n",
    "\n",
    "    results.to_csv(\n",
    "        file_path,\n",
    "        mode=\"a\",\n",
    "        header=not os.path.exists(file_path),\n",
    "        index=False\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f903e4e3",
   "metadata": {},
   "source": [
    "## 2. Dataset Description\n",
    "The dataset, sourced from Kaggle (ULB Machine Learning Group), contains credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions over a two-day period.\n",
    "\n",
    "The dataset is highly imbalanced, with only 492 fraudulent transactions, accounting for just 0.172% of the total data. Due to confidentiality reasons, the original features have been transformed using Principal Component Analysis (PCA).\n",
    "\n",
    "Features: 28 numerical features (V1 through V28) which are the principal components obtained with PCA.\n",
    "\n",
    "Un-transformed Features: Time (seconds elapsed between each transaction and the first transaction) and Amount (the transaction value).\n",
    "\n",
    "Target Variable: Class, a binary variable indicating whether the transaction is fraudulent (1 = yes, 0 = no)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87c3ec0f",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "To prepare the dataset for modeling, we will implement the following pipeline:\n",
    "\n",
    "- Scaling: While the PCA features (V1-V28) are already scaled, the Time and Amount features are not. We will apply a robust scaling method to the Amount feature to mitigate the influence of extreme transaction outliers.\n",
    "\n",
    "- Handling Class Imbalance: Because fraudulent transactions are extremely rare, training a model on the raw data will lead to bias. We will utilize sampling techniques (such as random undersampling of the majority class) to create a balanced subset of data for initial model training.\n",
    "\n",
    "- Splitting the Dataset: We will perform a train-test split using a stratified approach to ensure the original ratio of fraud-to-normal transactions is preserved across both the training and testing sets."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af174904-c3bc-41ab-8233-3d82b4acde69",
   "metadata": {},
   "source": [
    "### 3.1 Loading the Dataset\n",
    "\n",
    "We begin by downloading and loading the credit card fraud dataset from [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).\n",
    "\n",
    "We can utilize the provided code template to assist in the import."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0c06af50-4695-4aca-bf2e-6e8e5c7b84f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Path to dataset files: C:\\Users\\joshu\\.cache\\kagglehub\\datasets\\mlg-ulb\\creditcardfraud\\versions\\3\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>149.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>378.66</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>123.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>69.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
       "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
       "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
       "2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   \n",
       "3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   \n",
       "4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   \n",
       "\n",
       "        V26       V27       V28  Amount  Class  \n",
       "0 -0.189115  0.133558 -0.021053  149.62      0  \n",
       "1  0.125895 -0.008983  0.014724    2.69      0  \n",
       "2 -0.139097 -0.055353 -0.059752  378.66      0  \n",
       "3 -0.221929  0.062723  0.061458  123.50      0  \n",
       "4  0.502292  0.219422  0.215153   69.99      0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import kagglehub\n",
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# download dataset\n",
    "path = kagglehub.dataset_download(\"mlg-ulb/creditcardfraud\")\n",
    "\n",
    "print(\"Path to dataset files:\", path)\n",
    "\n",
    "# load the csv file\n",
    "csv_path = os.path.join(path, \"creditcard.csv\")\n",
    "df = pd.read_csv(csv_path)\n",
    "\n",
    "# preview data\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1012dfb3-01ec-410c-a469-abcd05b432d2",
   "metadata": {},
   "source": [
    "### 3.2 Defining Features and Labels\n",
    "\n",
    "We separate the dataset into a feature matrix (`X`) and target vector (`y`).\n",
    "\n",
    "We define our feature columns as all features **EXCEPT** our target `Class`. We can then easily use that definition to declare `X`, our feature matrix.\n",
    "\n",
    "From there, we can define `y` as our label vector using `class`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e1741f57-0915-48ac-8b9d-893df8cc6d5d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(284807, 30)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#define feature matrix using defined input columns\n",
    "# define features as all except Class (target)\n",
    "X = df.drop(\"Class\", axis=1)\n",
    "\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "54ac5183-0d2b-4870-8015-cb43ceb0f8c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         0\n",
       "50000     0\n",
       "100000    0\n",
       "150000    0\n",
       "200000    0\n",
       "250000    0\n",
       "Name: Class, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# define label vector\n",
    "y = df['Class']\n",
    "\n",
    "y[::50000]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0bc8cf2-6859-45f0-8756-8fb7dc19e7b9",
   "metadata": {},
   "source": [
    "### 3.3 Splitting the Dataset\n",
    "\n",
    "We split the dataset into training and testing sets using stratification to preserve the original class distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1a7b290c-7a73-4bb5-8aba-4da4fa59a62a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y,\n",
    "    test_size=0.2,\n",
    "    stratify=y,\n",
    "    random_state=RANDOM_STATE\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d43aa556-dd1b-4a50-9f06-98265cb54cd6",
   "metadata": {},
   "source": [
    "### 3.4 Class Distribution\n",
    "\n",
    "We visualize the distribution of fraudulent and non-fraudulent transactions to better understand the class imbalance in the dataset.\n",
    "\n",
    "Due to the extreme imbalance, where fraudulent transactions represent a very small fraction of the data, we annotate the counts to make the minority class more visible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2d57f4bb-ad29-4ddb-80be-c1d90e4d37fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlUAAAHHCAYAAACWQK1nAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAOmdJREFUeJzt3QmczfX+x/HPMNaxr8O1DKGxRcaaEv8mREq4IdcWSheFskxkSu7VP1f2pfLPcv/JUlH2rClGNJYQ/o1GEoYwRAzG+T8+3///dx7nzMLM+I5Zzuv5eJx75vx+3/M7v/Ob6Z637/I5fi6XyyUAAAC4Kznu7ukAAAAgVAEAAFhCTxUAAIAFhCoAAAALCFUAAAAWEKoAAAAsIFQBAABYQKgCAACwgFAFAABgAaEKyAaCgoKkV69ektW9+eab4ufnd09eq3nz5ubm2LJli3ntTz/99J68vv6+9PeWUXbu3Cm5c+eWX375RXzJjRs3pHz58jJz5syMPhVkQ4QqIBM7evSovPjii1K5cmXJmzevFCpUSJo2bSpTpkyRq1evSmY2b948E1Kcm55/2bJlpVWrVjJ16lT5448/rLzOyZMnTRjbu3evZDaZ+dxGjRolXbt2lYoVKybat2zZMnniiSekRIkSJnjp7+3ZZ5+VTZs2SVa/rrly5ZKhQ4fKP/7xD7l27Vq6nB98F6EKyKRWrVoltWvXliVLlki7du1k2rRpMn78eKlQoYIMGzZMXnnlFckKxo4dK//+979l1qxZMmjQILNt8ODB5r398MMPXm1Hjx6d6rCoH7BvvfVWqj9gv/rqK3NLT7c7tw8//FCOHDkiGUHPZ8OGDdK/f3+v7fpVsL1795YOHTpITEyMCR+zZ8+WAQMGyM8//yyPPfaYbN++XTJaWn/nDn2Pv//+uyxcuND6ucG3+Wf0CQBILDo6Wrp06WJ6EbR3oEyZMu59+gEXFRVlQldWoD0e9evXdz8OCwsz7+nJJ5+Up556Sg4dOiT58uUz+/z9/c0tPf3555+SP39+0wOTkbTHJKPMnTvXhPPGjRt7bZ84caLpYdTQ+95773kNxWrPlobj9P793AtFihSRli1bmvf6/PPPZ/TpIDtxAch0+vfv79L/PLdt25ai9hUrVnT17NnT/fjcuXOuV1991VWrVi1XQECAq2DBgq7WrVu79u7dm+i5U6dOddWoUcOVL18+V5EiRVwhISGujz/+2L3/0qVLrldeecW8Ru7cuV0lS5Z0hYaGuiIjI297TnPnzjXvYdeuXUnu/+c//2n2f/DBB+5t4eHhZpunr776ytW0aVNX4cKFzXupVq2aKywszOzbvHmzaZ/wpq+tHn30UVfNmjVd33//veuRRx4x71Hfi7NPbw7nWIsWLTLHL126tCt//vyudu3auY4fP37b6+3wPOadzk2fr8fxdPnyZdfQoUNd5cqVM9da3+uECRNct27d8mqnxxkwYIBr2bJl5v1pW/0drlmzxpUSFSpUcPXq1ctr259//ukqVqyYKzg42HXz5s0UHefo0aOuTp06uYoWLWqubaNGjVwrV65M8u8gOjraa7tzffTe8/rp+zl48KCrefPm5phly5Z1/ed//mei5yV3Xf/nf/7H1aFDB/P7y5Mnj+svf/mLq3Pnzq7Y2Fiv158yZYrLz8/P/LcC2JL1/8kBZEMrVqww86geeuihND1fh2qWL18uf/3rX6VSpUpmKOf999+XRx99VH788UczR8YZgnr55ZelU6dOZjhR55jokNx3330nzz33nGmjQ0Q6eXvgwIFSo0YNOXfunHz77bemh6levXppfo/du3eX119/3QzB9evXL8k2Bw8eND1aDzzwgBlGzJMnj+ml27Ztm9lfvXp1s33MmDHywgsvyCOPPGK2e143PV/tLdOev7/97W9SunTp256XzrXRHpoRI0bImTNnZPLkyRIaGmqGmpwetZRIybl50qykPXebN2+WPn36SN26dWXdunVmqPe3336TSZMmebXX38Hnn38uf//736VgwYJmnlrHjh3l+PHjUrx48WTPS4+lbRL+7vR458+fN71UOXPmvOP7078pfS/a86d/Q/qa8+fPN+9B/16eeeYZSYsLFy5I69atzRCkzuPSY+nvQoeL9fd4u+t6/fp1M2cvLi7ODDUHBgaa97ty5UqJjY2VwoULu18nJCTEXHMdztS/McAKa/EMgBUXL140//J++umnU/ychD0n165dc8XHx3u10Z4C/Zf72LFj3dv0NbRn4Ha0h0h7RVLrTj1VzrEffPDBZHuqJk2aZB6fPXs22WPo8T17Kjxpz4fumz17dpL7kuqp0p4N7Z1zLFmyxGzXno3U9FTd6dwS9lQtX77ctB03bpxXO+0J0h6VqKgo9zZtp71Tntv27dtn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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "counts = y.value_counts()\n",
    "\n",
    "ax = sns.barplot(x=counts.index, y=counts.values)\n",
    "\n",
    "for i, v in enumerate(counts.values):\n",
    "    ax.text(i, v, f\"{v:,}\", ha='center', va='bottom')\n",
    "\n",
    "plt.title(\"Class Distribution (Counts)\")\n",
    "plt.xlabel(\"Class\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b8179b4",
   "metadata": {},
   "source": [
    "## 4. Methods / Models Used\n",
    "To thoroughly investigate this dataset, we will implement a tiered modeling strategy, starting with baseline classical algorithms and progressing to advanced ensemble and deep learning techniques.\n",
    "\n",
    "A primary focus across all models will be addressing the extreme class imbalance. Alongside data sampling, we will heavily utilize model weight tuning (e.g., assigning higher class weights/penalties to fraudulent misclassifications) to force the algorithms to prioritize the minority class without needing to alter the underlying data distribution.\n",
    "\n",
    "### Classical Baselines\n",
    "\n",
    "- K-Nearest Neighbors (KNN): We will use KNN to establish a fundamental distance-based baseline for performance, allowing us to gauge the baseline complexity of separating the classes.\n",
    "\n",
    "- Linear Support Vector Machine (SVM): Because the dataset features (V1-V28) have already undergone Principal Component Analysis (PCA), we will test a Linear SVM to determine if the PCA projection created a linearly separable space for fraudulent and normal transactions.\n",
    "\n",
    "### Non-Linear & Ensemble Models\n",
    "\n",
    "- Non-Linear SVM: If the classes are not linearly separable, we will apply a Non-Linear SVM using the kernel trick (e.g., Radial Basis Function) to capture complex decision boundaries.\n",
    "\n",
    "- Random Forest & Gradient Boosted Trees: We will utilize powerful tree-based ensembles. Random Forest (bagging) will help reduce variance, while Gradient Boosting (boosting) will actively correct misclassifications. These models are highly resilient to imbalanced data and will serve as our primary classical classifiers.\n",
    "\n",
    "### Deep Learning\n",
    "\n",
    "- Artificial Neural Networks (ANN): To capture the most complex, underlying representations within the PCA features, we will design and train a multi-layer Artificial Neural Network. The network will be trained using backpropagation, with a focus on optimizing hidden layer architectures and activation functions. To prevent the network from simply predicting the majority \"normal\" class, we will implement custom loss functions and utilize weighted classes during the training phase.\n",
    "\n",
    "### Cross-Validation & Hyperparameter Tuning Strategy\n",
    "\n",
    "To ensure our models generalize to unseen data rather than memorizing the training set, rigorous cross-validation is mandatory across all tiers.\n",
    "\n",
    "- Stratified K-Fold Cross-Validation: Because the dataset contains only 0.172% fraudulent transactions, a standard random split would likely result in validation folds with zero fraud cases. We will strictly use Stratified K-Fold to guarantee that the exact ratio of fraud-to-normal transactions is preserved perfectly across every training and validation split.\n",
    "\n",
    "\n",
    "- Efficient Tuning: For computationally heavy models like the Non-Linear SVM and the ANN, we will pair our cross-validation with randomized search techniques on our sampled subsets to find optimal parameters without exceeding processing limits."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80c632b4-654c-4510-8b19-0fd14f569aa7",
   "metadata": {},
   "source": [
    "### 4.1 Logistic Regression\n",
    "\n",
    "We begin with Logistic Regression as a baseline model. To prevent data leakage and ensure proper evaluation, we use a pipeline that includes feature scaling and model training. Performance is evaluated using cross-validation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9d036b59-ae57-4e4a-8bd0-6ddaf296bfb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean F1: 0.7362314609436209\n",
      "Mean Recall: 0.6346153846153847\n",
      "Mean AUPRC: 0.7625674681864939\n"
     ]
    }
   ],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import cross_validate\n",
    "\n",
    "pipeline_lr = Pipeline([\n",
    "    (\"scaler\", StandardScaler()),\n",
    "    (\"model\", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))\n",
    "])\n",
    "\n",
    "scoring = {\n",
    "    \"f1\": \"f1\",\n",
    "    \"recall\": \"recall\",\n",
    "    \"auprc\": \"average_precision\"\n",
    "}\n",
    "\n",
    "scores_lr = cross_validate(\n",
    "    pipeline_lr,\n",
    "    X_train,\n",
    "    y_train,\n",
    "    cv=NUM_FOLDS,\n",
    "    scoring=scoring\n",
    ")\n",
    "\n",
    "print(\"Mean F1:\", scores_lr[\"test_f1\"].mean())\n",
    "print(\"Mean Recall:\", scores_lr[\"test_recall\"].mean())\n",
    "print(\"Mean AUPRC:\", scores_lr[\"test_auprc\"].mean())\n",
    "\n",
    "save_results(\"Logistic Regression\", scores_lr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f65c28ef-d795-45e8-a987-5ce3f87dcff2",
   "metadata": {},
   "source": [
    "### 4.2 KNN (K-Nearest Neighbors) Classifier\n",
    "\n",
    "We implement a K-Nearest Neighbors (KNN) classifier as another baseline model. KNN classifies a data point based on the majority label of its nearest neighbors in the feature space.\n",
    "\n",
    "Since KNN is sensitive to feature scaling, we incorporate standardization into a pipeline to ensure all features contribute equally to distance calculations. This prevents features with larger magnitudes from dominating the model.\n",
    "\n",
    "To determine the optimal number of neighbors (k), we perform hyperparameter tuning using GridSearchCV. We evaluate multiple values of k and select the one that maximizes the F1 score using cross-validation.\n",
    "\n",
    "After identifying the best k, we re-evaluate the model using cross-validation with multiple metrics, including F1 score, recall, and AUPRC, to ensure a fair comparison with other models.\n",
    "\n",
    "While KNN can be effective in capturing local patterns in the data, it may struggle with large datasets and highly imbalanced classes. Therefore, it is included primarily as a baseline for comparison against more advanced models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "bd96ce52-109e-4f1a-bec6-0a376f13cf9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "pipeline_knn = Pipeline([\n",
    "    (\"scaler\", StandardScaler()),\n",
    "    (\"model\", KNeighborsClassifier(weights=\"distance\"))\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "6a3b5a58-50dc-4fdd-9396-004e6e5d7254",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best k: {'model__n_neighbors': 3}\n",
      "Best score: 0.8479538534083704\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "param_grid_knn = {\n",
    "    \"model__n_neighbors\": [3, 5, 7, 11, 15]\n",
    "}\n",
    "\n",
    "\n",
    "grid_knn = GridSearchCV(\n",
    "    pipeline_knn,\n",
    "    param_grid_knn,\n",
    "    cv=NUM_FOLDS,\n",
    "    scoring=\"f1\",\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "grid_knn.fit(X_train, y_train)\n",
    "\n",
    "print(\"Best k:\", grid_knn.best_params_)\n",
    "print(\"Best score:\", grid_knn.best_score_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "0d019ba8-d425-43dd-ab8a-9193168a1275",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean F1: 0.8479538534083704\n",
      "Mean Recall: 0.7738461538461537\n",
      "Mean AUPRC: 0.8024839252013274\n"
     ]
    }
   ],
   "source": [
    "best_knn = grid_knn.best_estimator_\n",
    "\n",
    "scores_knn = cross_validate(\n",
    "    best_knn,\n",
    "    X_train,\n",
    "    y_train,\n",
    "    cv=NUM_FOLDS,\n",
    "    scoring={\n",
    "        \"f1\": \"f1\",\n",
    "        \"recall\": \"recall\",\n",
    "        \"auprc\": \"average_precision\"\n",
    "    }\n",
    ")\n",
    "\n",
    "print(\"Mean F1:\", scores_knn[\"test_f1\"].mean())\n",
    "print(\"Mean Recall:\", scores_knn[\"test_recall\"].mean())\n",
    "print(\"Mean AUPRC:\", scores_knn[\"test_auprc\"].mean())\n",
    "\n",
    "save_results(\"KNN\", scores_knn)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f78ac45",
   "metadata": {},
   "source": [
    "## 5. Results and Evaluation\n",
    "Because 99.8% of the data belongs to the \"Not Fraud\" class, standard accuracy is not a valid metric for this project. Instead, our evaluation will rely on:\n",
    "\n",
    "- AUPRC (Area Under the Precision-Recall Curve): The primary metric for highly imbalanced data.\n",
    "\n",
    "- Recall (Sensitivity): To measure our ability to catch as many actual frauds as possible.\n",
    "\n",
    "- F1-Score: To find the balance between precision and recall.\n",
    "\n",
    "(results are pending...)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cebc9023",
   "metadata": {},
   "source": [
    "## 6. Discussion\n",
    "Once models have been trained and evaluated, this section will discuss:\n",
    "\n",
    "- Which models performed best at detecting fraud and minimizing false positives.\n",
    "\n",
    "- The impact of data scaling and undersampling on model performance.\n",
    "\n",
    "- The computational challenges faced when running distance-based algorithms on large datasets.\n",
    "\n",
    "- Insights gained from the LLM-generated fraud explanations.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95f71f5b",
   "metadata": {},
   "source": [
    "## 7. Conclusion\n",
    "(To be completed: Summarize findings and possible future improvements for a production environment.)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b537ab11",
   "metadata": {},
   "source": [
    "References\n",
    "Credit Card Fraud Detection Dataset, Kaggle (MLG-ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
