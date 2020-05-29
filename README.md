# Optimized Conversion of Categorical and Numerical Features in Machine Learning Models
A study surveying categorical conversion methods for machine learning programs. Six datasets were provided, the largest containing over 40,000,000 tuples and 20 features. The task was to explore different strategies of converting categorical features into numerical features to be used as inputs to supervised learning algorithms. The goal was to determine which encoding techniques are the most effective and why. Methods were evaluated by the accuracy of predictive models, area under the receiver operating characteristics curve, and computation time of the conversion process. 

Problem was provided by Adobe Research.

## Paper Abstract
While some data have an explicit, numerical form, many other data, such as genderor  nationality,  do  not  typically  use  numbers  and  are  referred  to  as  categorical  data.Thus, machine learning algorithms need a way of representing categorical informationnumerically in order to be able to analyze them.  Our project specifically focuses on op-timizing the conversion of categorical features to a numerical form in order to maximizethe effectiveness of various machine learning models.  Of the methods we used, we foundthat Wide & Deep is the most effective model for datasets that contain high-cardinalityfeatures, as opposed to learned embedding and one-hot encoding.

### Background of Problem
Supervised learning models are the cornerstone of the many Machine Learning models we encounter in our lives every day. These start with a pair of values (x, y), where x is the vector of features and y is the label. For a mathematical model that maps x to y, we need x to be a vector of numbers. Unfortunately, for many problems of interest, the inputs are not numeric. For example, a person’s gender may take one of the following values Male, Female, Other, or Missing (Facebook allows 56 possible values for a person’s gender). Such features, with no inherent ordering of the values, are called categorical features. It is not always clear as to how we can convert such a feature into a numeric value.

### Research Goal
Using a number of real datasets, explore different strategies of converting categorical features into numeric features. All the datasets have a set of input features/covariates (categorical and numerical). The comparisons should be in terms of the following metrics, (1) Accuracy of a predictive model, (2) AUC of the ROC Curve, (3) Compute time of the conversion process.

### Datasets

| Name of Data Set | Size | Training Size | Testing Size | Features | Prediction Task | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Criteo Conversion | 15,898,883 | 70% | 30% | 9 numerical + 9 categorical | Click | - |
| Amazon Employee Access | 32,769 | 70% | 30% | 9 categorical | Is access appropriate for an employee? | - |
| Avazu Click Through Rate Prediction Rate Prediction | 40,428,968 | 50% | 50% | 20 categorical | Click on advertisement | Mobile app advertisement data |
| KDD 2009 | 50,000 | 70% | 30% | 189 categorical + 20 continuous | Two responses churn (16% positive) and appetency (2% positive) | The original data has 15K vars, kept all available categorical variables and only the top 20 (by abs(cor)) cont variables |
| US Census 1990 | 2,458,285 | 70% | 30% | 67 categorical | Artificial task created to predict if a person is married | This is US census data that has been obfuscated. A number of interesting variables are available. The task is a concocted one |
| Adult | 48,842 | 67% | 33% | 8 categorical | Predict if one's income is > 50k | - |

Authors: Wren Paris-Moe, Thomas Butler, Emily Liang, Andrea Stine

Note: only 2 of 6 datasets were under the size limit for uploading to GitHub
