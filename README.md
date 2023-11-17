# 97502025_Churning_Customers
# Customer Churn Prediction

This repository contains a Jupyter Notebook (`CustomerChurn_Assignment3.ipynb`) focused on predicting customer churn using machine learning techniques. The notebook covers various stages of the data science pipeline, including data collection, explorative data analysis, data preprocessing, feature selection, model building, and evaluation.

## Table of Contents
- [Data Collection](#data-collection)
- [Explorative Data Analysis](#explorative-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Selecting Relevant Features](#selecting-relevant-features)
- [Scaling for App](#scaling-for-app)
- [Model Building](#model-building)
- [Retrain and Retest](#retrain-and-retest)
- [Saving the Model](#saving-the-model)

## Data Collection

The dataset is loaded from a CSV file (`CustomerChurn_dataset.csv`). It contains information about customers, including contract type, payment method, tech support, online security, and more.

## Explorative Data Analysis

EDA is performed to gain insights into the dataset. Visualizations include a pie chart showing the likelihood of customer churn and count plots for various categorical features.

## Data Preprocessing

Data preprocessing involves handling missing values, encoding categorical variables, and scaling numerical features. The notebook drops the 'customerID' column and encodes the target variable using Label Encoding.

## Selecting Relevant Features

Feature importance is determined using a Random Forest Classifier, and the top 9 features are selected for model training.

## Scaling for App

The selected features are scaled using StandardScaler for use in the machine learning model.

## Model Building

A neural network model is built using the Keras library. The model is trained and evaluated on the dataset. Hyperparameter tuning is performed using GridSearchCV.

## Retrain and Retest

The best model is retrained and retested to ensure its performance on the test set.

## Saving the Model

The final trained and optimized model is saved as 'deploy.h5'. Additionally, the StandardScaler used for scaling is saved as 'scaled_model.pkl'.

Feel free to use this repository as a guide for understanding and implementing customer churn prediction in your projects.

Find below the link to a video that shows the deployment. Enjoy!
https://drive.google.com/file/d/1ThBwhQb7DmkBXV_KlwZj_XRVgCFlbNWt/view?usp=sharing


