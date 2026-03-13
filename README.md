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
-The dataset contains 782 patients, which represents a satisfactory sample size for a medical study.
It includes 56 variables, indicating a rich dataset with many clinical characteristics.
-The dataset is well structured, with explanatory variables (features) and target variables (labels) separated, which facilitates data analysis and machine learning preparation.
-Among the variables:
    * 39 variables are of type object:
These are categorical (qualitative) variables representing symptoms, diagnoses, or clinical characteristics expressed as text or categories.
    * 17 variables are of type float64:
These are numerical (quantitative) variables, such as age, weight, or biological measurements.
-The 56 variables include several types of medical information:
   Demographic data
Age/Sex/Height/Weight/BMI
   Clinical scores
Alvarado Score/Pediatric Appendicitis Score
   Symptoms
Migratory Pain/Nausea/Loss of Appetite/Dysuria
   Biological examinations
WBC Count/Neutrophil Percentage/CRP/RBC in Urine
   Imaging results
Appendix on Ultrasound (US)/Appendix Diameter/Free Fluids
   Target variables
Diagnosis/Management/Severity

## 4- Exploratory data analysis

### Missing Values Analysis

 The dataset contains missing values in most variables, which is common in real clinical datasets where not all tests are performed for every patient.
 -Out of the 56 total variables, all except two (likely identifiers) contain missing values.

 **Variables with a very high number of missing values**

Variable                 	Missing	                 Percentage
Abscess_Location	           769	                      98.34%
Gynecological_Findings	     756	                      96.68%
Conglomerate_of_Bowel_Loops  739	                      94.50%
Segmented_Neutrophils	     728	                      93.09%
Ileus	                       722	                      92.33%
Perfusion	                 719	                      91.94%
Enteritis	                 716	                      91.56%
Appendicolith	              713                     	 91.18%
Coprostasis	                 711	                      90.92%
-The very high percentage of missing values (often greater than 90%) can be explained by the fact that these advanced examinations are only performed when the standard ultrasound shows abnormalities.
**Consequence:**

These variables cannot be directly used in a predictive model, because the large number of missing values would significantly affect the reliability of the model.

**Critical variables with very few missing values**

Variable                 	Missing	                   Percentage
Diagnosis	                 2	                         0.26%
Age	                       1	                         0.13%
Management	                 1	                         0.13%
Severity	                    1	                         0.13%
Sex	                       2	                         0.26%
-The essential variables (such as diagnosis, age, and sex) are almost complete.
-Only 2 patients out of 782 (0.26%) have a missing diagnosis, meaning that these rows can be safely removed without significant data loss.
-Similarly, age is missing for only one patient (0.13%), which has minimal impact on the dataset.
### Target Variable Analysis

-The analysis of the target variable Diagnosis reveals critical insights into the dataset's class distribution. As shown in the notebook, the dataset contains 782 patients with the following distribution:

Diagnosis	                           Count	                          Percentage
Appendicitis	                         463	                            59.4%
No Appendicitis	                      317	                            40.6%

The visualization includes both a bar chart showing the absolute counts (463 vs 317 patients) and a pie chart displaying the percentage distribution (59.4% vs 40.6%).
***Class Balance Assessment***

The dataset presents a moderate class imbalance with an 18.8% difference between classes (59.4% - 40.6% = 18.8%). This exceeds the 10% threshold typically considered for balanced datasets.
***Impact on Modeling***

This imbalance has important implications for the machine learning phase:
-Accuracy alone will not be a reliable metric, as a model predicting "appendicitis" for all cases would achieve 59.4% accuracy
-F1-score, precision, and recall must be monitored, particularly for the minority class (no appendicitis)
-Class weighting techniques (class_weight='balanced') should be considered in models like Random Forest or Logistic Regression
-Confusion matrix analysis will be essential to evaluate performance on both classes
### OUTLIER ANALYSIS ###

Outlier analysis is a crucial step in any machine learning project, particularly in a medical context. This analysis aims to identify extreme values that could:
   -Distort descriptive statistics (mean, standard deviation)
   -Disproportionately influence machine learning models
   -Represent rare but important clinical cases to preserve
   -Reveal data entry errors in the dataset
***Visualization with Boxplots***

Boxplots were created for each numerical variable to visualize:
 -The median (center line)
 -Quartiles (Q1 and Q3) forming the box
 -The interquartile range (IQR)
 -Whiskers representing normal value limits
 -Points beyond whiskers (potential outliers)
 ***Quantification with the IQR Method***

 The IQR (Interquartile Range) method was applied with the standard threshold of 1.5:

 IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
Outlier = any value < lower bound OR > upper bound

***Quantitative Results***

Variable	                         Number of Outliers	                             Percentage
CRP	                                   85	                                        10.87%
Length_of_Stay	                          44	                                        5.63%
RDW	                                   23	                                        2.94%
BMI	                                   23	                                        2.94%
Hemoglobin	                             16	                                        2.05%
RBC_Count	                             14	                                        1.79%
Body_Temperature	                       12	                                        1.53%
Height	                                12	                                        1.53%
Appendix_Diameter	                       10	                                        1.28%
Thrombocyte_Count	                       9	                                        1.15%
Total: 14 variables present outliers, with rates ranging from 1.15% to 10.87%.
***Medical Interpretation of Results***

-CRP (C-Reactive Protein) - 10.87% outliers
  Normal values: generally < 5 mg/L
  Observed values: up to 365 mg/L
  Interpretation: CRP is an inflammatory marker that can reach very high values during severe infections. These "outliers" actually represent clinically severe cases that are essential to retain
-Length_of_Stay - 5.63% outliers
  Observed values: up to 28 days
  Interpretation: Extended stays correspond to complicated cases (perforated appendicitis, peritonitis) or post-operative complications.
-BMI (Body Mass Index) - 2.94% outliers
  Observed values: up to 38.16
  Interpretation: These high values correspond to cases of severe obesity in children, clinically possible and important to retain.
-Body_Temperature - 1.53% outliers
  Observed values: up to 40.2°C
  Interpretation: High fevers are important clinical signs of severe infection.
   ***PROMPT ENGINEERING: Understanding Boxplots and IQR Method***
        **Context of the Request**

During this analysis, I wanted to deepen my understanding of boxplots and the IQR method to better interpret the results and explain them clearly in this documentation
 -Initial Prompt (Question asked to AI):
 "why is it necessary to analyze outliers and what are boxplots and the IQR method"
        **AI Response**

The AI provided a detailed explanation structured as follows:
   -Definition of outliers:
"Outliers are extreme values significantly different from others. For example, if most patients are between 5 and 15 years old, a 50-year-old patient would be an outlier."
3 main reasons to analyze outliers:
Data entry errors: impossible values to correct
Rare but real clinical cases: important to retain
Impact on ML models: outliers can mislead algorithms
 -Explanation of boxplots:
"A boxplot summarizes a distribution with 5 numbers: minimum, Q1, median, Q3, maximum. The box represents the interquartile range (IQR) containing 50% of the data."
 -Explanation of IQR method:

IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
OUTLIER = Any value < lower bound OR > upper bound
     **Changes and Improvements Made**

Following this explanation, I was able to:
-Understand the complementarity between boxplots (visualization) and IQR (quantification)
-Correctly interpret the generated boxplot results
-Validate the relevance of the 1.5 threshold as a statistical standard
-Distinguish between "impossible" outliers (errors) and "clinically significant" outliers
    **Application to Our Analysis**

Thanks to this deeper understanding, I was able to:
-Classify outliers by type (biological vs potential errors)
-Justify retaining CRP outliers (severe cases)
-Identify variables requiring special attention
-Clearly document results for the team
  
### Correlation Analysis###

As part of our exploratory data analysis on appendicitis, we dedicated a significant section to studying correlations between different variables in the dataset. This analysis aims to identify potential linear relationships between numerical variables, which is crucial for understanding the underlying structure of the data and for preparing potential subsequent modeling. The main objective was to detect possible multicollinearity issues that could affect the performance of predictive models.
***Methodology of Correlation Analysis***

Our approach to analyzing correlations unfolded in two distinct but complementary phases. First, we generated a complete correlation matrix including all numerical variables in the dataset. This matrix was visualized as a heatmap, allowing rapid visual inspection of relationships between variables. The choice of a color scale ranging from blue (negative correlations) to red (positive correlations) facilitates instant identification of strongly correlated pairs.

In the second phase, we systematically identified pairs of variables with an absolute correlation greater than 0.7, a threshold generally considered indicative of a strong correlation. This automated approach allowed us to precisely extract the six most strongly correlated variable pairs for in-depth analysis.
***Results of Strong Correlation Identification***

The analysis revealed six pairs of variables with correlations above the 0.7 threshold. These pairs are, in descending order of correlation:
 -Height and Age with a correlation of 0.865
 -Weight and Body Mass Index (BMI) with a correlation of 0.859
 -Paediatric Appendicitis Score and Alvarado Score with a correlation of 0.832
 -Weight and Height with a correlation of 0.830
 -Weight and Age with a correlation of 0.766
 -Neutrophil Percentage and Alvarado Score with a correlation of 0.701
***Visualization and In-Depth Analysis***

To better understand these relationships, we created specific visualizations for the four most strongly correlated pairs. Each graph presents a scatter plot accompanied by a trend line, allowing visual appreciation of the strength and nature of the linear relationship.

The graphs clearly show the data dispersion and visually confirm the calculated correlation coefficients. The presence of trend lines facilitates identifying the direction of the relationship and helps detect potential outliers that might influence the correlation.
***Decision-Making and Recommendations***

Faced with these identified correlations, we developed a multi-level classification system to guide decision-making regarding the treatment of these variables in future analyses:
 -Correlations above 0.95: Considered critical, requiring mandatory removal of one variable
 -Correlations between 0.9 and 0.95: Very strong, strongly recommending removal of one variable
 -Correlations between 0.85 and 0.9: Strong, requiring careful monitoring in linear models
 -Correlations between 0.8 and 0.85: Moderate, acceptable but with recommendation to use robust models
 -Correlations below 0.8: Acceptable, requiring no specific action

Based on this framework, we classified the identified pairs as follows:

 -The Height-Age and Weight-BMI pairs (correlations > 0.85) are classified as "strong" and require monitoring in linear models

 -The Paediatric_Appendicitis_Score-Alvarado_Score and Weight-Height pairs (correlations between 0.8 and 0.85) are classified as "moderate" and can be retained, with a recommendation to use robust models such as random forests

 -The Weight-Age and Neutrophil_Percentage-Alvarado_Score pairs (correlations < 0.8) are considered acceptable and can be retained without specific action
  
  




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

### Models Trained
We trained and compared four machine learning models on the processed 
and balanced dataset prepared by Sara (data_processed_and_balanced.xlsx), 
which contained 796 patients and 39 features, with a perfectly balanced 
target variable (398 appendicitis / 398 no appendicitis).

We split the data into 80% for training (636 patients) and 20% for 
testing (160 patients) using stratified splitting to preserve the 
class distribution.

### Performance Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| SVM | 0.7937 | 0.8073 | 0.9123 | 0.6500 | 0.7591 |
| Random Forest | 0.9812 | 0.9995 | 1.0000 | 0.9625 | 0.9809 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CatBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Final Model Selection
We automatically selected **LightGBM** as the best model based on the 
highest ROC-AUC score of 1.0. In a medical context, we prioritized 
Recall over Precision because missing a real appendicitis case 
(false negative) is far more dangerous than a false alarm 
(false positive). LightGBM achieved a perfect Recall of 1.0, 
meaning we missed zero real appendicitis cases in the test set.

The best model is saved at: `models/best_model.pkl`  
The feature names are saved at: `models/feature_names.pkl`

### Critical Question — Was the dataset balanced?
The original dataset contained 463 appendicitis cases (59%) and 
317 no appendicitis cases (41%), which represented a moderate 
imbalance. We applied oversampling to balance both classes to 
398 each. After balancing, all model performances improved 
significantly — LightGBM and CatBoost both reached a perfect 
score of 1.0. However, this perfect score may partially indicate 
overfitting due to the oversampling technique, which should be 
analyzed further with cross-validation.

### Critical Question — Which ML model performed best?
LightGBM performed best with a ROC-AUC of 1.0, Accuracy of 1.0, 
Precision of 1.0, Recall of 1.0, and F1-Score of 1.0 on the 
balanced test set. Random Forest was a close second with a 
ROC-AUC of 0.9995.

---

## Prompt Engineering
### Task Selected: Universal data preparation function `auto_prepare_data()`

**Exact prompt used:**
> "Write a universal Python function called auto_prepare_data(filepath, 
> target_col=None) that works with ANY dataset. It should:
> - Automatically detect CSV or Excel format by file extension
> - Auto-detect the target column if not specified by searching for 
>   keywords like 'diagnosis', 'target', 'label', 'outcome'
> - Handle missing values using median imputation
> - Encode categorical columns automatically using LabelEncoder
> - Map text labels like 'appendicitis' to 1 and 'no appendicitis' 
>   to 0 using explicit string mapping after converting to lowercase
> - Return X features and y target as integers ready for ML training"

**Result obtained:**
The generated function worked with any dataset format and 
automatically handled all preprocessing steps. It correctly 
detected the Diagnosis column and encoded the target variable.

**What worked well:**
Being very specific about the expected string mapping behavior 
(lowercase conversion before mapping) was the key to solving the 
encoding bug where 'appendicitis' was not being mapped to 1 
correctly due to pandas StringArray dtype differences in newer 
pandas versions.

**What we would improve:**
We would specify in the prompt to always convert the target column 
using `.astype(str).str.strip().str.lower()` before any mapping 
operation, which would have avoided the encoding bug we encountered 
during testing where the distribution showed `{0: 780}` instead 
of the expected `{1: 463, 0: 317}`.

---

## 7- Model Evaluation

We evaluated the best saved model by generating three diagnostic 
outputs stored in the `results/` folder.

### Confusion Matrix
The confusion matrix shows that LightGBM correctly identified all 
appendicitis and no appendicitis cases in the test set with zero 
false negatives — which is critical in a medical context where 
missing a real appendicitis case can have serious consequences.

### ROC Curve
The ROC curve confirms the perfect discrimination ability of the 
model with an AUC of 1.0, meaning we perfectly separate the two 
classes across all classification thresholds.

### Classification Report
The classification report shows perfect precision, recall, and 
F1-score of 1.0 for both classes on the balanced test set.

Results saved in:
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/classification_report.txt`

### CI/CD Pipeline
We set up a GitHub Actions CI/CD pipeline in 
`.github/workflows/ci.yml` that automatically installs all 
dependencies and runs the pytest test suite on every push to main, 
ensuring the codebase remains stable and reproducible throughout 
the project.

## 8- SHAP analysis 

## 9- Web interface 

## 10- Project Structure

    pediatric-appendicitis/
    ├── .github/
    │   └── workflows/
    │       └── ci.yml
    ├── app/
    │   └── app.py
    ├── data/
    │   ├── app_data.xlsx
    │   └── data_processed_and_balanced.xlsx
    ├── models/
    │   ├── best_model.pkl
    │   └── feature_names.pkl
    ├── notebooks/
    │   └── eda.ipynb
    ├── results/
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── classification_report.txt
    ├── src/
    │   ├── data_processing.py
    │   ├── train_model.py
    │   └── evaluate_model.py
    ├── tests/
    │   └── test_data_processing.py
    ├── requirements.txt
    └── README.md

## 11- Installation

```bash
git clone https://github.com/1inaSis/pediatric-appendicitis.git
cd pediatric-appendicitis
pip install -r requirements.txt
```

---

## 12- Model training
```bash
python src/train_model.py
```

This will:
- Load the processed dataset from `data/data_processed_and_balanced.xlsx`
- Train and compare SVM, Random Forest, LightGBM, and CatBoost
- Automatically select the best model based on ROC-AUC
- Save the best model to `models/best_model.pkl`
- Save feature names to `models/feature_names.pkl`

To evaluate the model and generate result plots:
```bash
python src/evaluate_model.py
```

---
## 13- App launch
```bash
streamlit run app/app.py
```

The app will open at: `http://localhost:8501`

---