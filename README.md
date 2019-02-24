# Kaggle-Titanic

*A machine learning model predicting the survival of Titanic passengers*

*Language:* R

The repository contains an R script for predicting the survival status of the Titanic passengers listed in the *test.csv* file of the famous [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). The machine learning model used to make these predictions is trained on the *train.csv* file data. Both *test.csv* and *train.csv* are available on the [competition web site](https://www.kaggle.com/c/titanic/data).

The script acomplishes the following goals:

* imputing missing and erroneous values present in the original data files; 
* introducing some new features;
* creating an [XGBoost](https://xgboost.readthedocs.io/en/latest/#) model for making the survival status predictions;
* generating a csv submission file to be submitted on the [competition web site](https://www.kaggle.com/c/titanic).

The leaderboard (LB) score that you can reach with this script is **0.80382**.

The repository contains the following files:

1. 'Titanic_caret_xgb_80382.R' -- the R script file.
2. 'README.md' -- this file.
