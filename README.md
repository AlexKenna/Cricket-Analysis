# Predicting Player Performance in Cricket
![Heading Image](https://github.com/AlexKenna/Cricket-Predicting-Player-Performance/blob/main/img/ODI_Batting.jpg?raw=true)

## Introduction
Effective player selection is critical for team performance in cricket. This is especially true at the international level, where the best players are selected to represent their nation, largely based on their perceived potential. However, without an accurate method for predicting the performance of batters, selections are often made based on traditional metrics deemed important with little empirical evidence, such as domestic batting averages. This project aims to develop a method for predicting batter performance at the international level based on their domestic careers. In doing so, we will identify and quantify the most important metrics for determining international success. This is achieved through the derivation of a Random Forest model that uses domestic data to predict the batting average of a player in One Day International cricket. The model is fitted based on a subset of domestic metrics determined to be most important through feature selection techniques including recursive feature elimination and clustering. This enables the development of a model that is explainable, efficient, and accurate. Experimentally, the results support the idea that player performance is largely determined by the traditional metrics currently used for player selection. They also provide insight into lesser-known metrics, such as which domestic formats produce better international batsmen and how a batter’s contribution to their team translates to international performance. These results allow us to identify the features of a batter’s domestic career that have the most significant influence on international performance. This has practical implications in team selection, talent identification and player comparison.


## Overview
This project consisted of six main steps:

1. Data Exploration
2. Data Cleaning
3. Data Summarisation
4. Feature Extraction
5. Hyperparameter Tuning
6. Model Testing

Each of these steps are explored in separate notebooks in the notebooks directory.


## Method

### 1. Data Exploration
An exploration into the raw data was conducted to gain an understanding of the general data structure. The raw data consisted of 1.1B+ datapoints.

### 2. Data Cleaning
Data cleaning occurred according to the findings of the data exploration phase. The following points summarise the cleaning steps that were taken:

* Removal of redundant data
* Handling missing data
* Removal of outliers
* Identification of eligible batters for modelling

After cleaning, the dataset was reduced to ~550M datapoints.

### 3. Data Summarisation
The cleaned data was summarised into a 130 feature dataset consisting of 62 batters. For domestic T20, One-Day and Test formats, the following categories of features were determined:

* Batter attributes (e.g., Name)
* Matches played (e.g., Number of One-Day matches)
* Wickets (e.g., Frequency out bowled)
* Runs (e.g., High score)
* Milestones (e.g., Frequency of 100 runs)
* Batting Position (e.g., Number of wickets fallen when a batter enters)
* Batting style (e.g., Strike rate)
* Team contribution (e.g., Run contributions to their team)

### 4. Feature Extraction
From the ~130 feature summary, the 20 most important features were determined. The following feature selection techniques were explored:

* Mean Decrease in Impurity importance
* Permutation importance (clustered and unclustered)
* Covariance matrix (feature-feature and feature-target correlation)
* Sequential feature selection (forward and backward)
* Recursive feature elimination

### 5. Hyperparameter Tuning
Once the feature set was reduced, the hyperparameters of the model were tuned to optimise performance. A random search was performed first to reduce the search space of possible values. Then, a grid search was performed to determine the optimal hyperparameters. The following hyperparameters were investigated:

* Number of trees
* Maximum number of features to consider at each split
* Maximum depth of each tree
* Minimum number of samples to split an internal node
* Minimum number of samples allowed at each leaf
* Bootstrapping

### 6. Model Testing
Once the final model was obtained, its performance was tested on a validation set according to three measures:

* Coefficient of Determination
* Mean Absolute Error
* Root Mean Squared Error

The results of these measures are outlined in the next section.


## Results

### Feature Importance
Using various feature selection techniques, we were able to determine the most important features for model performance. A summary of the most important features is outlined below.

| Domestic Features | 
| :--- | 
| Batting Average |
| Team Run Contribution | 
| Average Number of Balls Faced | 
| 50 Rate | 
| Batting Position | 
| Team High Scorer Rate | 
| High Score | 
| Start Rate (1-49 Runs) | 
| Not Out Percentage | 


### Error Metrics
To determine the accuracy of the model, three quantitative metrics were used and are recorded below.

| Metric | Score |
| --- | ----------- |
| Coefficient of Determination | 0.788 |
| Mean Absolute Error | 3.684 |
| Root Mean Squared Error | 4.722 |


### Visual Comparison
To visualise model accuracy, we have plotted each player's true and predicted batting averages in a scatter plot. The close proximity of the points to the main diagonal suggests that the model is reasonably accurate.

![Comparison of True and Predicted Averages](https://github.com/AlexKenna/Cricket-Predicting-Player-Performance/blob/main/img/Comparison_of_True_and_Predicted_Average.jpg?raw=true)


## Acknowledgements
I would like to thank Prof. Chris Drovandi (QUT), Dr. Kate Saunders (QUT), Thomas Body (Cricket Australia) and Charles Evans (Queensland Cricket) for their support and guidance during this project.

The following resources were also helpful:

1. Breiman, L. (2001). Random Forests. Berkeley: University of California.
2. Brownlee, J. (2016, April 22). Bagging and Random Forest Ensemble Algorithms for Machine Learning. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/
3. Brownlee, J. (2020, April 20). How to Develop a Random Forest Ensemble in Python. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/random-forest-ensemble-in-python/
4. Keboola. (2020, September 17). The Ultimate Guide to Random Forest Regression. Retrieved from Keboola: https://www.keboola.com/blog/random-forest-regression
5. Koehrsen, W. (2018, August 21). An Implementation and Explanation of the Random Forest in Python. Retrieved from Towards Data Science: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
6. Koehrsen, W. (2018, January 10). Hyperparameter Tuning the Random Forest in Python. Retrieved from Towards Data Science: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
7. Mithrakumar, M. (2019, November 12). How to tune a Decision Tree? Retrieved from Towards Data Science: https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
8. Passi, K., & Pandey, N. (2018). Increased Prediction Accuracy In The Game Of Cricket Using Machine Learning. International Journal of Data Mining and Knowledge Management Process, 19-36.
9. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., . . . Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2825-2830.
10. Stevenson, O. G., & Brewer, B. J. (2019). Modelling Career Trajectories of Cricket Players Using Gaussian Processes. Auckland: Springer Proceedings in Mathematics & Statistics.
