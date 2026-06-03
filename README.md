# Stroke Risk Prediction Using Machine Learning

This project uses machine learning to predict whether a patient is likely to have a stroke based on healthcare, demographic and lifestyle-related information.

The main idea was not only to build a prediction model, but also to understand how different machine learning models behave when the dataset is highly imbalanced. In this dataset, most patients did not have a stroke, while only a small number of patients had a stroke. Because of that, accuracy alone was not enough to judge the models.

This project was completed for the CSE422 Artificial Intelligence course.

## Project Overview

The dataset contains patient information such as age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status and stroke status.

The target variable is `stroke`, where:

* `0` means the patient did not have a stroke
* `1` means the patient had a stroke

Since the output has only two possible classes, this is a binary classification problem.

## Dataset

The dataset contains:

* 5110 data points
* 12 columns in total
* 10 useful input features after removing the `id` column and target column
* One target column: `stroke`

The dataset includes both numerical and categorical features.

Numerical features include:

* `age`
* `avg_glucose_level`
* `bmi`

Categorical features include:

* `gender`
* `ever_married`
* `work_type`
* `Residence_type`
* `smoking_status`

The dataset is highly imbalanced. Most records belong to the no-stroke class, while only a small portion belong to the stroke class.

## Preprocessing Steps

Before training the models, several preprocessing steps were applied:

1. Removed the `id` column because it is only a unique identifier and does not help with prediction.
2. Handled missing BMI values using median imputation.
3. Encoded categorical features using one-hot encoding.
4. Scaled numerical features using StandardScaler.
5. Used stratified train-test splitting to preserve the class distribution in both training and testing sets.

The dataset was split into:

* 80% training data
* 20% testing data

Scaling was fitted only on the training data to avoid data leakage.

## Models Used

Several supervised machine learning models were trained and tested:

* Neural Network
* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes

K-Means clustering was also applied to treat the problem as an unsupervised learning task and analyze patient risk groups based on feature similarity.

## Model Results

The models were compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* AUC score

The results showed that accuracy can be misleading for this dataset. For example, KNN achieved high accuracy, but it failed to detect actual stroke cases properly because the dataset was highly imbalanced.

Logistic Regression gave the best overall supervised performance. It had the highest AUC score and the best F1-score among the supervised models. Neural Network also performed well after applying class weights. Naive Bayes had very high recall but produced too many false positives.

## Best Model

The best overall model was Logistic Regression.

It performed better because it gave a more balanced result compared to the other models. It was able to identify many actual stroke cases while also maintaining better overall classification performance than the other models.

## K-Means Clustering

K-Means clustering was used for unsupervised patient group analysis.

The model divided patients into two clusters based on feature similarity. After comparing the clusters with the actual stroke labels for interpretation, one cluster contained more stroke cases than the other. This suggests that clustering can help identify higher-risk patient groups.

However, K-Means should not be treated as a direct stroke prediction model because it does not use the target label during training.

## Key Takeaways

This project showed that healthcare classification problems can be difficult when the dataset is imbalanced.

A model may look good based on accuracy but still fail to detect the minority class. For stroke prediction, recall, precision, F1-score, confusion matrix, ROC curve and AUC score are more useful than accuracy alone.

The project also showed that preprocessing steps such as missing value handling, encoding, scaling and stratified splitting are very important before training machine learning models.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/stroke-risk-prediction-ml.git
```

2. Go inside the project folder:

```bash
cd stroke-risk-prediction-ml
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook stroke_prediction_analysis.ipynb
```

5. Run the notebook cells from top to bottom.

## Files

```text
stroke-risk-prediction-ml/
│
├── README.md
├── healthcare-dataset-stroke-data.csv
├── stroke_prediction_analysis.ipynb
├── requirements.txt
└── report/
    └── CSE422_Section_06_Group_10_Project_Report.pdf
```

## Note

This project is for educational purposes only. It should not be used as a real medical diagnosis system. Stroke risk prediction in real life requires proper clinical validation, expert review and reliable medical data.
