# Cricket-Predicting-Player-Performance
![alt text](https://github.com/AlexKenna/Cricket-Predicting-Player-Performance/blob/main/img/ODI_Batting.jpg?raw=true)

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


## Results


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
