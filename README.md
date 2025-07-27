## Project Overview

This project demonstrates the end-to-end process of building and evaluating a machine learning model to predict customer churn for a telecom company. Leveraging Python, Pandas, and Scikit-learn, this project covers essential data science steps from initial data loading and extensive preprocessing to model training (using a Random Forest Classifier) and comprehensive performance evaluation, culminating in identifying key factors influencing churn.

This is a side project developed to enrich my resume and showcase practical skills in data analysis, machine learning, and data visualization.

**Developed with guidance from Google's Gemini AI.**

## Table of Contents

- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Step 1: Project Setup and Data Acquisition](#step-1-project-setup-and-data-acquisition)
- [Step 2: Load and Initial Explore the Dataset](#step-2-load-and-initial-explore-the-dataset)
- [Step 3: Data Cleaning and Preprocessing](#step-3-data-cleaning-and-preprocessing)
- [Step 4: Feature Scaling and Data Splitting](#step-4-feature-scaling-and-data-splitting)
- [Step 5: Model Training - Random Forest Classifier](#step-5-model-training---random-forest-classifier)
- [Step 6: Model Evaluation](#step-6-model-evaluation)
- [Step 7: Feature Importance Analysis](#step-7-feature-importance-analysis)
- [Key Project Visuals](#key-project-visuals)
- [How to Run This Project](#how-to-run-this-project)
- [Contact](#contact)

## Step 1: Project Setup and Data Acquisition

This section handles the initial setup of the Google Colab environment and the loading of the Telco Customer Churn dataset from Google Drive. The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) was obtained from Kaggle.

## Step 2: Load and Initial Explore the Dataset

Here, the dataset is loaded into a Pandas DataFrame. An initial exploration is performed to understand the data's structure, identify data types, and check for basic statistics and class distribution of the target variable ('Churn'). This step helps in identifying initial data quality issues.

## Step 3: Data Cleaning and Preprocessing

This is a critical step to prepare the raw data for machine learning models. It involves:
- Handling missing values (specifically in 'TotalCharges' by filling with the median).
- Correcting data types (converting 'TotalCharges' from object to numeric).
- Encoding categorical variables into a numerical format suitable for models (using binary mapping for 'Yes'/'No' columns and One-Hot Encoding for multi-category features).
- Removing irrelevant columns like 'customerID'.

## Step 4: Feature Scaling and Data Splitting

In this step, numerical features are scaled using `StandardScaler` to standardize their range, which helps optimize model performance. The dataset is then split into training and testing sets (80% train, 20% test) to ensure an unbiased evaluation of the model's ability to generalize to unseen data. Stratified splitting is used to maintain the proportion of churners in both the training and testing sets, crucial for imbalanced datasets.

## Step 5: Model Training - Random Forest Classifier

A Random Forest Classifier, an ensemble machine learning model, is trained on the prepared training data. This model learns the complex patterns and relationships between various customer features and their churn status, building multiple decision trees and aggregating their predictions.

## Step 6: Model Evaluation

The performance of the trained Random Forest model is rigorously evaluated using key metrics on the unseen test data. Metrics calculated include:
- **Accuracy:** Overall correctness of predictions.
- **Precision:** Proportion of positive predictions that were actually correct.
- **Recall:** Proportion of actual positives that were correctly identified.
- **F1-Score:** Harmonic mean of precision and recall, useful for imbalanced datasets.
- **ROC AUC Score:** Measures the model's ability to distinguish between churners and non-churners.
- A **Classification Report** provides per-class metrics, and a **Confusion Matrix** visualizes the true positives, true negatives, false positives, and false negatives.

## Step 7: Feature Importance Analysis

This step identifies the most influential features that the Random Forest model used to predict customer churn. Understanding feature importance provides valuable business insights into what truly drives customer attrition (e.g., contract type, monthly charges, tenure), which can inform targeted retention strategies and business decisions.

## Key Project Visuals

Here are some of the key outputs and visualizations from the project, showcasing data preprocessing, model performance, and feature insights:

### Processed Data Head

![Processed Data Head](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction/blob/main/screenshots/processed_data_head.png?raw=true)

### Processed Data Info

![Processed Data Info](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction/blob/main/screenshots/processed_data_info.png?raw=true)

### Model Performance Metrics

![Model Performance Metrics](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction/blob/main/screenshots/model_performance_metrics.png?raw=true)

### Confusion Matrix

![Confusion Matrix](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction/blob/main/screenshots/confusion_matrix_plot.png?raw=true)

### Feature Importance

![Feature Importance](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction/blob/main/screenshots/feature_importance_plot.png?raw=true)

## How to Run This Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction.git](https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction.git)
    ```
2.  **Open in Google Colab:**
    * Go to [Google Colab](https://colab.research.google.com/).
    * Click `File` > `Open notebook` > `GitHub`.
    * Paste the repository URL: `https://github.com/Dinu-Sreekumar/Customer-Churn-Prediction` and hit Enter.
    * Select the `Customer_Churn_Prediction.ipynb` notebook.
3.  **Mount Google Drive:**
    * In the Colab notebook, run the first code cell to mount your Google Drive.
    * Ensure the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset is placed in `My Drive/Colab Notebooks/CustomerChurn/`.
4.  **Run All Cells:** Execute all cells sequentially (`Runtime` -> `Run all`) to reproduce the analysis and model training.

## Contact

Feel free to connect with me for any questions, feedback, or collaborations on this project!

* **LinkedIn:** [Dinu Sreekumar](https://www.linkedin.com/in/dinu-sreekumar)
