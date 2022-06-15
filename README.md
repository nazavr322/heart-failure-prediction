# About
Small project about compairing different machine learning models on [heart failure clinical records dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).
The most accurate one is deployed using FastAPI and Heroku. You can try it out [here](https://nazavr322.github.io/heart-failure-prediction/)

## Demonstration
![](https://media.giphy.com/media/WXMKkMQdylaWAUUGAX/giphy.gif)

# Project Overview
This project covers following topics:
 - Data visualisation
 - Statistical significance tests (Parametric and Nonparametric) / Gini Impurity Reduction
 - Overview and comparison of different classification metrics (F-beta score, ROC AUC, PR AUC)
 - Overview and comparison of different ML algorithms for classification (LogReg, SVC, RandomForestClassifier etc.)
 - Building API for the model with FastAPI, deploy using Heroku, building web-page with HTML, CSS and Brython

You can see all the steps in the `test.ipynb` file.

## Comparison results
In the table below all models are sorted by the F-0.5 Score and PR AUC.

| | F-0.5 Score | ROC AUC | PR AUC |
| :---: |   :---:    |  :---:  | :---:  |
| **LogisticRegression** | 0.615301 | 0.793269 | 0.679791 | 
| **SVC** | 0.600834 | 0.795072 | 0.677686 |    
| **MultinomialNB** | 0.579207 | 0.783053 | 0.663107 |  
| **RandomForestClassifier** | 0.568594 | 0.826022 | 0.657011 |
| **KNeighborsClassifier** | 0.509483 | 0.790264 | 0.607543 |
| **DecisionTreeClassifier** | 0.490124 | 0.770733 | 0.597404 |
