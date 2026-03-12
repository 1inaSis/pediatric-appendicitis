# Pediatric Appendicitis Diagnosis with Machine Learning

## 1- Introduction

Acute appendicitis is one of the most common causes of abdominal pain requiring emergency surgery in children. However, diagnosing appendicitis can be challenging because many symptoms overlap with other abdominal conditions. Inaccurate diagnosis may lead either to unnecessary surgeries or to delayed treatment, both of which can have serious consequences for patients.

This project aims to develop a machine learning system capable of assisting in the diagnosis of pediatric appendicitis. Using clinical data from the Regensburg Pediatric Appendicitis Dataset, the project explores how data-driven models can help predict the likelihood of appendicitis based on patient characteristics, laboratory results, and clinical observations.

In addition to building predictive models, the project also focuses on model interpretability. Medical applications require transparency and trust, so it is important not only to obtain accurate predictions but also to understand the factors influencing the model’s decisions.

Finally, the project integrates the trained model into a simple and user-friendly interface, allowing doctors to input patient information and obtain a prediction along with an explanation of the result. 

## 2- Objectives of the project

The main objective of this project is to develop a machine learning system that can assist in the diagnosis of pediatric appendicitis using clinical data.

More specifically, the project aims to:

* Analyze the pediatric appendicitis dataset and understand the structure and characteristics of the clinical data.
* Perform data preprocessing, including handling missing values, cleaning the dataset, and preparing the data for machine learning models.
* Train and evaluate several machine learning models in order to identify the most effective approach for predicting appendicitis.
* Compare the performance of different models using appropriate evaluation metrics.
* Improve the interpretability of the predictions using explainability techniques such as SHAP.
* Develop a simple and interactive application that allows doctors to input patient information and obtain a prediction.
* Ensure that the results are understandable and transparent, which is particularly important in medical decision-support systems.

## 3- Dataset 

## 4- Exploratory data analysis

## 5- Data Processing 

Before training the machine learning models, we cleaned the dataset and prepared it through several preprocessing steps. The goal of this stage is to transform the raw clinical data into a reliable and structured dataset suitable for modeling.

### Loading the Dataset

We first loaded the clinical dataset  from an Excel file using Python and the pandas library. This allows the data to be manipulated in a structured DataFrame format.

### Handling Missing Values

The dataset initially contains several missing values. First, we analyzed the number of missing values in each variable . Columns with too many missing values (higher number than 390 which is half the number of patients ) or unreliable clinical measurements are removed. In addition, rows corresponding to patients with missing values in essential diagnostic variables (such as age, sex, body temperature, or key laboratory results) are also removed.

### Removing Irrelevant Variables

Several variables related to specific ultrasound observations or rare medical conditions were removed because they contained too many missing values or were not sufficiently reliable for the analysis.

### Outlier Treatment

To handle extreme values in the dataset, we chose to apply **winsorization** instead of the **Interquartile Range (IQR) method**.

The IQR method typically removes observations that fall outside a specific range. While this approach is effective for detecting outliers, it may lead to the loss of potentially useful medical information, especially in relatively small datasets.

In contrast, winsorization limits the influence of extreme values by capping them at predefined percentiles (in our case, the 5th and 95th percentiles) rather than removing the observations entirely. This approach allows us to reduce the impact of extreme values while preserving all patient records.

Because the dataset contains clinical measurements where extreme values may still be medically meaningful, winsorization was considered a more appropriate and conservative strategy.


### Feature and Target Separation

Then, we divided the dataset into two categories:

* **Input features :** clinical variables describing patient characteristics and test results.
* **Target variable :** the diagnosis indicating whether the patient has appendicitis.

### Class Imbalance Handling

The dataset initially contains an imbalance between the two diagnostic classes (appendicitis vs. non-appendicitis). To address this issue, we applied an oversampling technique to balance the dataset by increasing the number of samples in the minority class. This ensures that both classes contain the same number of observations.

### Memory Optimization

Finally, we optimized to reduce memory usage. Numerical columns are converted to smaller data types (e.g., int64 to int16 or float64 to float32) whenever possible, which improves computational efficiency without affecting the data integrity.

### Saving the Processed Dataset

After preprocessing and balancing, we saved the final dataset as a new Excel file. This processed dataset is then used for training and evaluating the machine learning models.

### Test data-processing

In order to be sure that our data was correctly processed, we implemented a test which verifies that the dataset isn't empty, that it doesn't have too much missing values and that it is alanced. The results of those tests were successful and it allowed us therefore to push it in the repository.

### Test memory optimization

We implemented a simple test to make sure that the function **optimize_memory()** really reduces the memory capacity of the dataset and doesn't do something to increase it.

### Prompt engineering

- During the data preprocessing stage, we used prompt engineering  to explore different strategies for handling outliers in the dataset.

An initial prompt was used to ask for possible methods to treat extreme values in numerical medical variables. The suggested approaches included the **Interquartile Range (IQR) method**, winsorization, and standard statistical filtering.

After reviewing the results, one limitation of the IQR-based approach was identified: it removes observations that fall outside a defined range. In a medical dataset with a limited number of patients, removing rows could result in the loss of potentially important clinical information.

To address this issue, we refined the prompt to focus on methods that preserve all observations while still reducing the influence of extreme values. Based on this analysis, we selected **winsorization**  because it limits extreme values without removing patient records, making it more suitable for this dataset.

- We used another prompt to explore strategies for handling class imbalance in the dataset. The suggested approaches included undersampling, oversampling, and synthetic data generation methods such as SMOTE.

After analyzing these options, we selected oversampling as the most appropriate solution for this project. Undersampling would have reduced the number of available observations, while more complex techniques like SMOTE were not necessary given the moderate imbalance of the dataset. Oversampling allowed us to balance the classes while preserving all the original information.


## 6- ML Models 

## 7- Model evaluation 

## 8- SHAP analysis 

## 9- Web interface 

## 10- Project structure

## 11- Installation

## 12- Model training

## 13- App launch