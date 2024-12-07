# Churn Modeling Classification with ANN

This project implements a classification model to predict customer churn using an Artificial Neural Network (ANN). The model is built using TensorFlow and Keras, and the performance is evaluated using K-fold cross-validation. Additionally, TensorBoard is integrated for visualizing the training process.

## Features
- **Churn Prediction**: The model predicts whether a customer will churn based on demographic, account, and usage features.
- **Cross-Validation**: K-fold cross-validation is used to evaluate model performance and avoid overfitting.
- **TensorBoard Visualization**: Real-time visualization of the training process, including loss and accuracy curves.

## Requirements

Ensure that you have the following libraries installed:

- Python 
- Keras
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- TensorBoard
- streamlit
- seaborn


## Dataset

The dataset contains information about customers' demographics, financial information, and behavior. The target variable `Exited` indicates whether the customer has exited (1) or not (0). The dataset features include:

| Feature            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `RowNumber`        | Row number in the dataset                                                   |
| `CustomerId`       | Unique identifier for each customer                                          |
| `Surname`          | Customer's surname                                                           |
| `CreditScore`      | Customer's credit score                                                      |
| `Geography`        | Country of the customer (e.g., France, Spain, Germany)                       |
| `Gender`           | Customer's gender (Male/Female)                                              |
| `Age`              | Age of the customer                                                           |
| `Tenure`           | Number of years the customer has been with the company                       |
| `Balance`          | Average yearly balance in euros                                              |
| `NumOfProducts`    | Number of products the customer has subscribed to                            |
| `HasCrCard`        | Whether the customer has a credit card (1 = Yes, 0 = No)                     |
| `IsActiveMember`   | Whether the customer is an active member (1 = Yes, 0 = No)                   |
| `EstimatedSalary`  | Estimated yearly salary of the customer in euros                             |
| `Exited`           | Target variable indicating whether the customer exited (1 = Yes, 0 = No)     |

Sample rows from the dataset:

| RowNumber | CustomerId | Surname | CreditScore | Geography | Gender | Age | Tenure | Balance | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
|-----------|------------|---------|-------------|-----------|--------|-----|--------|---------|---------------|-----------|----------------|-----------------|--------|
| 1         | 15634602   | Hargrave | 619         | France    | Female | 42  | 2      | 0       | 1             | 1         | 1              | 101348.88       | 1      |
| 2         | 15647311   | Hill     | 608         | Spain     | Female | 41  | 1      | 83807.86| 1             | 0         | 1              | 112542.58       | 0      |
| 3         | 15619304   | Onio     | 502         | France    | Female | 42  | 8      | 159660.8| 3             | 1         | 0              | 113931.57       | 1      |
| 4         | 15701354   | Boni     | 699         | France    | Female | 39  | 1      | 0       | 2             | 0         | 0              | 93826.63        | 0      |

## Model Description

The model used in this project is a **Artificial Neural Network (ANN)** designed to predict whether a customer will exit the company based on their demographic, financial, and usage data. Below is an overview of the architecture:

### ANN Architecture:
- **Input Layer**: Corresponds to the number of features in the dataset.
- **Hidden Layers**: One or more hidden layers with **ReLU** activation functions. These layers learn complex patterns in the data.
- **Output Layer**: A **sigmoid** activation function to predict a binary outcome (whether the customer will exit or not).

The model is compiled using the **Adam** optimizer and **binary cross-entropy** as the loss function, suitable for binary classification problems.

### Hyperparameter Tuning:
Hyperparameters such as the number of epochs, neurons per layer, and the number of hidden layers are tuned using **GridSearchCV** to optimize the model's performance.

### Cross-Validation:
To ensure the model generalizes well, we employ **3-fold cross-validation** to evaluate the model's performance and avoid overfitting.

### TensorBoard Integration:
To monitor the training process, we have integrated **TensorBoard** for visualizing metrics such as loss and accuracy during the training of the model.

## Steps Involved:

1. **Preprocessing & EDA**: Clean the data, handle missing values, and encode categorical variables. Visualization of all numerical variables
2. **Model Development**: Create the ANN with appropriate layers and activations.
3. **Training**: Train the model using the training dataset.
4. **Hyperparameter Tuning**: Use **GridSearchCV** to find the optimal hyperparameters.
5. **Cross-Validation**: Perform **K-fold cross-validation** for a robust evaluation.
6. **Evaluation on validation data**: Assess the model's performance using accuracy, precision, recall, and F1-score.
7. **Deployment on streamlit**: This app is been deployed on streamlit and from there it predicts for live data entered.

