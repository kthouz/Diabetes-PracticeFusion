# Diabetes-PracticeFusion
Predicting diabetic patients using EHR provide by Practice Fusion &amp; Kaggle

In this project, I used data provided by Kaggle and Practice Fusion. Multiple sets of patients information, transcripts, diagnosis, medications and patients allergies were provided in a [kaggle competition](https://www.kaggle.com/c/pf2012-diabetes) that was closed in 2012. I should note that this is my first data science project in healthcare using EHR.

**Objective:** Build a model to predict positive diabetes

**Procedure**

### 1. Data cleaning
The cleaning of data consisted in:


#### a. Text mining medication names
There is a combination of medications that can weaken a patient and sometimes triggering symptoms of diabetes mellitus. In this notebook, I am going to text mine medication names provided in Practice Fusion datasets with the goal to find components of medications that are linked to risk of diabetes type 2. Results from this notebook will be used to build the [predictive model](https://github.com/kthouz/Diabetes-PracticeFusion/blob/master/3_EDA%20%26%20Model%20Building.ipynb)  [(continue reading ...)](https://github.com/kthouz/Diabetes-PracticeFusion/blob/master/1_Text%20Mining%20-%20Medication%20%26%20Transcript.ipynb)

#### b. Cleaning transcripts and diagnosis
- validating data types
- imputing missing values
- dropping outliers
[(continue reading ...)](https://github.com/kthouz/Diabetes-PracticeFusion/blob/master/2_Transcripts%20%26%20Diagnosis%20-%20Cleaning.ipynb)

### 2. EDA, Model building and validation
The process consists in:

1. Loading cleaned data and building one large dataset. containing all information
2. Carrying an exploratory data analysis. I will generate visualization that make it easy to see correlations between independent and dependent variables. In this section, statistical tests will also be used to validate separability of the dependent variable among different groups of the independent variables
3. Building a predictive model. Three different Gradient Boost ML algorithms will be tuned and to optimize the best parameters for final predictions
4. Validating the models

[(continue reading ...)](https://github.com/kthouz/Diabetes-PracticeFusion/blob/master/3_EDA%20%26%20Model%20Building.ipynb)


### 5. Summary and recommendations

Three models were built by tuning the Gradient Boost ML algorithm using clean data from patients information, transcripts, diagnosis and medications. An ensemble model was tuned through a grid search of optimum parameters (number of trees, max depth, ...) and 5 fold cross-validation using using data from patients info, transcripts and diagnosis to produce ROC-AUC 
of 0.7537. A tremandous improvement was achieved by using tf-idf data obtained from medications name. This new model produced  ROC-AUC 0.9939.

Clearly, medication names are very important to predict potential diabetes patients if historical medical records of patients are available. Althought an accuracy of 0.9939 is very good, I would recommend using tf-idf on n-grams instead of single terms. This would give better intuitive understanding about the population. The model built using n-gram would also be robust for future medications which are not necessary present in the current dataset
