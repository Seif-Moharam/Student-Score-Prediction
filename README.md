# Student Score Prediction

## Overview
This project predicts students' exam scores based on factors like study hours, sleep quality, and participation using a neural network. It includes data preprocessing, feature selection, and performance visualization.

## Dataset
[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

## Key Steps
1. Data cleaning (handled missing values)
2. Feature encoding (converted categorical to numerical)
3. Feature selection (removed low-correlation features)
4. Data splitting (80% train, 20% test)
5. Feature standardization

## Training
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Early Stopping (patience=10)
Metrics: MAE

## Training Loss Curve
![alt text](https://github.com/Seif-Moharam/Student-Score-Prediction/blob/master/Training%20Loss%20Curve.png)

## Results:
![alt text](https://github.com/Seif-Moharam/Student-Score-Prediction/blob/master/True%20vs%20Predicted%20Scores%20Scatter%20Plot.png)

## Requirements:

Python 3.12

pandas

tensorflow

scikit-learn

matplotlib
