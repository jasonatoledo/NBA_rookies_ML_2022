# -*- coding: utf-8 -*-
"""
Jason Toledo
Class: CS 677
Date: 04/23/2022
Homework Problem # Final Project
This script reads in a CSV file for NBA player data, uses 4 models and selects 
the one with the best accuracy. An unseen data file containing rookie player 
data and predicts whether they will be in the league 5+ years and exports an 
updated CSV file to the user.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from prettytable import PrettyTable

# read file into dataframe
df = pd.read_csv("nba_players.csv")

# drop unnamed column and name, no value
df = df.iloc[:,2:]

# drop any duplicate values
df = df.drop_duplicates().reset_index(drop=True)

# plot correlation
plt.figure(figsize=(15,11))
sns.heatmap(df.corr(), annot=True, cmap='rocket')
plt.title('NBA Players Rookie Statistics, Target: 5 Years in League',\
    fontsize=15)
plt.show()

# create X & Y variables
X = df.iloc[:,:-1].values
Y = df[["target_5yrs"]].values

# create scaler object
scaler = StandardScaler()

# scale X
X = scaler.fit_transform(X)

# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.80,\
    random_state=46)

# create logistic regression classifier
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train.ravel())
Y_pred = log_reg_classifier.predict(X_test)

# training score
training_score = log_reg_classifier.score(X_train, Y_train)
print("\nThe 1st Logistic Regression Training Score:",
"%.4f" % training_score)

# get accuracy
LR1_accuracy = accuracy_score(Y_test, Y_pred)
print('The 1st Logistic Regression Accuracy:', 
"%.4f" % LR1_accuracy)

# create confusion matrix to check metrics
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nLR1 confusion matrix:\n",conf_matrix)

# assign variables
LR1_TP = conf_matrix[1][1]
LR1_FN = conf_matrix[0][1]
LR1_TN = conf_matrix[0][0]
LR1_FP = conf_matrix[1][0]

# get true pos/neg values
LR1_TPR = round(LR1_TP/(LR1_TP + LR1_FN),4)
LR1_TNR = round(LR1_TN/(LR1_TN + LR1_FP),4)

# print RMSE value
LR1_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('LR1 Root Mean Squared Error:', 
"%.4f" % np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# print feature importances
feature_imp = log_reg_classifier.coef_[0]
# summarize feature importance
print("\nLogistic Regression Feature Importances:\n")
for i,j in enumerate(feature_imp):
    print('Feature: %0d, Score: %.5f' % (i+1,j))

# try logistic regression again, this time removing features
df2 = df.copy()

"""dropping these columns because they are either a percentage and rebounds is 
the sum of the oreb and dreb values""" 
df2 = df2.drop(columns=["fg","reb","ft","3p"]).reset_index(drop=True)

# check new correlation values
df2.corr()

# create X & Y variables
X = df2.iloc[:,:-1].values
Y = df2[["target_5yrs"]].values

# create scaler object
scaler = StandardScaler()

# scale X
X = scaler.fit_transform(X)

# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.80,\
    random_state=46)

# create logistic regression classifier
log_reg_classifier = LogisticRegression(penalty = 'l2', C = 1, 
    solver='newton-cg' ,random_state=46)
log_reg_classifier.fit(X_train, Y_train.ravel())
Y_pred = log_reg_classifier.predict(X_test)

# training score
training_score = log_reg_classifier.score(X_train, Y_train)
print("\nThe 2nd Logistic Regression Training Score:",
"%.4f" % training_score)

# get accuracy
LR2_accuracy = accuracy_score(Y_test, Y_pred)
print('The 2nd Logistic Regression Accuracy:', 
"%.4f" % LR2_accuracy)

# create a confusion matrix and print
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nLR2 confusion matrix:\n",conf_matrix)

# assign variables
LR2_TP = conf_matrix[1][1]
LR2_FN = conf_matrix[0][1]
LR2_TN = conf_matrix[0][0]
LR2_FP = conf_matrix[1][0]

# get true pos/neg values
LR2_TPR = round(LR2_TP/(LR2_TP + LR2_FN),4)
LR2_TNR = round(LR2_TN/(LR2_TN + LR2_FP),4)

# print RMSE value
LR2_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('LR2 Root Mean Squared Error:', 
"%.4f" % np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

feature_imp = log_reg_classifier.coef_[0]
# summarize feature importance
print("Logistic Regression Feature Importances:\n")
for i,j in enumerate(feature_imp):
    print('Feature: %0d, Score: %.5f' % (i,j))

# plot feature importance
plt.bar([x for x in range(len(feature_imp))], feature_imp)
plt.xticks(np.arange(0,len(df2.columns)-1))
plt.title("Logistic Regression Feature Importance Bar Chart")
plt.show()

# create X & Y variables
X = df2.iloc[:,:-1].values
Y = df2[["target_5yrs"]].values

# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.80,\
    random_state=46)

# create N & d lists and combined list with all combinations
N = list(range(1,31))
d = list(range(1,11))

# put error rates for all combinations into a df
error_rates = []
data = []

for i in N:
    for j in d:
        RFC = RandomForestClassifier(n_estimators=i, max_depth=j, \
            criterion ='entropy', random_state=46)
        RFC.fit(X_train, Y_train.ravel())
        Y_pred = RFC.predict(X_test)
        error_rate = np.mean(Y_pred!=Y_test)  
        error_rates.append(error_rate)
        data.append([i,j,error_rate])

# create dataframe object
my_df = pd.DataFrame(columns=["N", "error_rate", "d"])
N_list = [item[0] for item in data]
d_list = [item[1] for item in data]
error_rate_list = [item[2] for item in data]
my_df["N"] = N_list
my_df["d"] = d_list
my_df["error_rate"] = error_rate_list

# get best N & d value for RFC
error_min = my_df["error_rate"].min()
finder = my_df.loc[my_df["error_rate"] == error_min]
best_n_d = [int(finder.iloc[0]["N"]), int(finder.iloc[0]["d"])]
print("\nThe best N, d combo is:", best_n_d)

# create model based on best N & d combination
RFC = RandomForestClassifier(n_estimators=best_n_d[0], max_depth=best_n_d[1], \
    criterion ='entropy')
RFC.fit(X_train, Y_train.ravel())
Y_pred = RFC.predict(X_test)

# calculate best accuracy value
RFC_accuracy = accuracy_score(Y_test, Y_pred)
print("The accuracy for the best combination of N and d is:", \
    "%.4f" % RFC_accuracy)

# create and display confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nBest N & d confusion matrix printout:\n",conf_matrix)

# print RMSE value
RFC_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('\nRFC Root Mean Squared Error:', 
"%.4f" % np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# create a confusion matrix based on best N & d combination
conf_matrix = confusion_matrix(Y_test, Y_pred)

# assign variables
RFC_TP = conf_matrix[1][1]
RFC_FN = conf_matrix[0][1]
RFC_TN = conf_matrix[0][0]
RFC_FP = conf_matrix[1][0]

# get true pos/neg values
RFC_TPR = round(RFC_TP/(RFC_TP + RFC_FN),4)
RFC_TNR = round(RFC_TN/(RFC_TN + RFC_FP),4)

# XGBoost create X & Y variables on encoded data
X = df2.iloc[:,:-1].values
Y = df2[["target_5yrs"]].values

# scale X
X = scaler.fit_transform(X)

# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.80,\
    random_state=46)

# fit model no training data
XGB = XGBClassifier(n_estimators=100, max_depth=3, random_state=46, \
    min_child_weight=0.5, gamma=0.1, booster="gbtree", learning_rate=0.01,\
        subsample=1, objective='binary:logistic', n_jobs=4, \
            use_label_encoder=False)
XGB.fit(X_train, Y_train.ravel())

# make predictions for test data
Y_pred = XGB.predict(X_test)

# evaluate predictions
XG_accuracy = accuracy_score(Y_test, Y_pred)
print("\nThe XGBoost Accuracy is:", "%.4f" % XG_accuracy)

# create a confusion matrix based on best N & d combination
conf_matrix = confusion_matrix(Y_test, Y_pred)

# assign variables
XG_TP = conf_matrix[1][1]
XG_FN = conf_matrix[0][1]
XG_TN = conf_matrix[0][0]
XG_FP = conf_matrix[1][0]

# get true pos/neg values
XG_TPR = round(XG_TP/(XG_TP + XG_FN),4)
XG_TNR = round(XG_TN/(XG_TN + XG_FP),4)

# print RMSE value
XGB_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('\nXGBoost Root Mean Squared Error:', 
"%.4f" % np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# try NB classifier supervised model
NB_classifier = GaussianNB().fit(X_train, Y_train.ravel())
Y_pred = NB_classifier.predict(X_test)

# reshape test & prediction data
Y_test = Y_test.reshape(-1,1)
Y_pred = Y_pred.reshape(-1,1)

# split data into train & test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, \
    random_state=46)

# create NB classifier object, train and get metrics
NB_classifier = GaussianNB().fit(X_train, Y_train.ravel())
Y_pred = NB_classifier.predict(X_test)
NB_accuracy = accuracy_score(Y_test, Y_pred)

# print accuracy value
print("\nThe accuracy using NB is:", "%.4f" % NB_accuracy)

# create a confusion matrix based on best N & d combination
conf_matrix = confusion_matrix(Y_test, Y_pred)

# assign variables
NB_TP = conf_matrix[1][1]
NB_FN = conf_matrix[0][1]
NB_TN = conf_matrix[0][0]
NB_FP = conf_matrix[1][0]

# get true pos/neg values
NB_TPR = NB_TP/(NB_TP + NB_FN)
NB_TNR = NB_TN/(NB_TN + NB_FP)

# print RMSE value
NB_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('\nNaive-Bayes Root Mean Squared Error:', 
"%.4f" % np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# summarize the results of all of the models' performance
my_table = PrettyTable(["Model", "TP", "FP", "TN","FN", "accuracy","TPR","TNR"])
my_table.add_row(["Logistic Regression 1", LR1_TP, LR1_FP, LR1_TN, LR1_FN, \
    "%.4f" % LR1_accuracy, "%.4f" % LR1_TPR, "%.4f" % LR1_TNR])
my_table.add_row(["Logistic Regression 2", LR2_TP, LR2_FP, LR2_TN, LR2_FN, \
    "%.4f" % LR2_accuracy, "%.4f" % LR2_TPR, "%.4f" % LR2_TNR])
my_table.add_row(["Random Forest", RFC_TP, RFC_FP, RFC_TN, RFC_FN, \
    "%.4f" % RFC_accuracy, "%.4f" % RFC_TPR, "%.4f" % RFC_TNR])
my_table.add_row(["XGBoost", XG_TP, XG_FP, XG_TN, XG_FN, "%.4f" % XG_accuracy, \
    "%.4f" % XG_TPR, "%.4f" % XG_TNR])
my_table.add_row(["Naive-Bayes", NB_TP, NB_FP, NB_TN, NB_FN, \
    "%.4f" % NB_accuracy, "%.4f" % NB_TPR, "%.4f" % NB_TNR])
print("\n------------ Summary Table ------------ \n", my_table)

# create RMSE table
my_table = PrettyTable(["Model", "RMSE"])
my_table.add_row(["Logistic Regression 1", "%.4f" % LR1_RMSE])
my_table.add_row(["Logistic Regression 2", "%.4f" % LR2_RMSE])
my_table.add_row(["Random Forest", "%.4f" % RFC_RMSE])
my_table.add_row(["XGBoost", "%.4f" % XGB_RMSE])
my_table.add_row(["Naive-Bayes", "%.4f" % NB_RMSE])
print("\n----------- RMSE Table ----------- \n", my_table)

# select best model based on accuracy
accuracies = [LR1_accuracy, LR2_accuracy, RFC_accuracy, XG_accuracy, \
    NB_accuracy]
print("\nThe highest accuracy of the models is:", "%.4f" % max(accuracies))

# Use the LR2 model to predict unseen data outcomes
# create X & Y variables
X = df2.iloc[:,:-1].values
Y = df2[["target_5yrs"]].values

# create scaler object
scaler = StandardScaler()

# scale X
X = scaler.fit_transform(X)

# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.80,\
    random_state=46)


# read in unseen NBA rookies data
rookies_df = pd.read_csv("nba_rookies.csv")

# remove unneeded columns
rookies_df_clean = rookies_df.iloc[:,3:]

# predict target_5yrs using model
R_X = rookies_df_clean.iloc[:,:].values

# predict rookie 5 year outcomes with worst performing model
Rookie_Pred_NB = NB_classifier.predict(R_X)

# take predictions and add back to rookies_df as a column
Rookie_Pred_Vals = list(Rookie_Pred_NB)
rookies_df["5_yr_prediction_NB"] = Rookie_Pred_Vals
# check value counts of prediction
print("\nUsing Worst Performing Model, Naive-Bayes: \n")
print("The size of the rookie class list is:", len(rookies_df))
print("The number of rookies who will play 5+ years is:",
sum(rookies_df["5_yr_prediction_NB"]==1))
print("The number of rookies who will not play 5+ years is:",
sum(rookies_df["5_yr_prediction_NB"]==0))

# create logistic regression classifier
log_reg_classifier = LogisticRegression(penalty = 'l2', C = 1, \
    solver='newton-cg' ,random_state=46)
log_reg_classifier.fit(X_train, Y_train.ravel())
Y_pred = log_reg_classifier.predict(X_test)

# predict rookie 5 year outcomes with best performing model
Rookie_Pred_LR = log_reg_classifier.predict(R_X)

# take predictions and add back to rookies_df as a column
Rookie_Pred_Vals = list(Rookie_Pred_LR)
rookies_df["5_yr_prediction_LR"] = Rookie_Pred_Vals
# check value counts of prediction
print("\nUsing Best Performing Model, Logistic Regression: \n")
print("The size of the rookie class list is:", len(rookies_df))
print("The number of rookies who will play 5+ years is:",
sum(rookies_df["5_yr_prediction_LR"]==1))
print("The number of rookies who will not play 5+ years is:",
sum(rookies_df["5_yr_prediction_LR"]==0))

# create export of rookies dfs
rookies_df.to_csv("nba_rookie_predictions.csv", index=False)
print("\nThe updated Rookies CSV file has been created!")



