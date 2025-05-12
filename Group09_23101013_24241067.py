import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


payment_fraud_df = pd.read_csv('./online_payments_fraud_detection_dataset.csv')

print(payment_fraud_df)

# EDA

print(f"Shape of the dataset {payment_fraud_df.shape}")
print(f"The dataset has {payment_fraud_df.shape[0]} rows and {payment_fraud_df.shape[1]} columns")

payment_fraud_df.info()

features_names = payment_fraud_df.columns.tolist()
print(f"There are total {len(features_names)} features")
print(features_names)

numerical_features_data = payment_fraud_df.select_dtypes(include='number')
numerical_features_name = numerical_features_data.columns.tolist()

print(f"There are total {len(numerical_features_name)} numerical features")
print(numerical_features_name)

categorical_features_data = payment_fraud_df.select_dtypes(include='object')
categorical_features_name_list = categorical_features_data.columns.tolist()

print(f"There are total {len(categorical_features_name_list)} categorical features")
print(categorical_features_name_list)

categorical_features_data.describe().T

numerical_features_data.var()

numerical_features_data.skew()

numerical_features_data.hist(figsize=(15,8),bins=20)
plt.show()

#data preprocessing
initial_null_values = payment_fraud_df.isnull().sum()
initial_null_values

#As There are no null values in the dataset, No imputation or row/column removal is needed for missing data

print(f"Only {payment_fraud_df['isFlaggedFraud'].sum()} payments are flagged as fraud out of {payment_fraud_df.shape[0]} transactions!\nSo this feature (isFlaggedFraud) may leak target")

print(f'"nameOrig" and "nameDest" are also not useful for modeling, and these may introduce noise')

# deleting less relevant columns

payment_fraud_df = payment_fraud_df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

# current state of the dataframe
payment_fraud_df

# after deleting (Categorical data)

categorical_features_data = payment_fraud_df.select_dtypes(include='object')
categorical_features_name_list = categorical_features_data.columns.tolist()

print(f"There are total {len(categorical_features_name_list)} categorical features")
print(categorical_features_name_list)

unique_categorical_data_count = categorical_features_data.nunique()
print(unique_categorical_data_count)

# as there is only 1 column, this loop will run only once!
for column in categorical_features_name_list:
    plt.title(f"'{column}' distribution")
    categorical_features_data[column].value_counts().plot(kind="bar", ylabel='count')
    plt.show()


#Encoding categorical data

label_encoder = LabelEncoder()

payment_fraud_df['type'] = label_encoder.fit_transform(payment_fraud_df['type'])

payment_fraud_df

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Categorical values are encoded like this: ")
for key, value in label_mapping.items():
    print(f"{key} = {value}")


df_encoded = pd.get_dummies(payment_fraud_df, columns=['type'], drop_first=True)
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Features")
plt.show()


# "oldbalanceOrg" and "newbalanceOrig" these two attributes are in perfect correlation (1.00). This indicates these two values are identical.
# Similarly, "oldbalanceDest" and "newbalanceDest" also possess almost perfect correlation (0.98), indicating they are almost the same.



# removing the dependencies by dropping one of the columns from each pair
payment_fraud_df = payment_fraud_df.drop(columns=["newbalanceOrig", "newbalanceDest"])

df_encoded = pd.get_dummies(payment_fraud_df, columns=['type'], drop_first=True)
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Features")
plt.show()



output_feature = 'isFraud'
payment_fraud_df[output_feature].value_counts()


output_count_list = payment_fraud_df[output_feature].value_counts().tolist()

total_outputs = payment_fraud_df.shape[0]

labels = ["Not Fraud", "Fraud"]
percentage = []

for value in output_count_list:
    percentage.append((value/total_outputs)*100)
print(f"Percentage: {percentage}")

barChart_labels = ["Not Fraud", "Fraud"]
notFraud, isFraud = payment_fraud_df[output_feature].value_counts().tolist()

plt.bar(barChart_labels, percentage)
plt.ylabel("Percentage")
plt.title('Bar Chart of Heart Disease Count')

# There is a massive class imbalance in the dataset. This biases machine learning models towards the majority class. 
# The model becomes good at predicting the majority class but bad at the minority one.</br> Overall accuracy can be misleadingly high. 
# Detection of the rare, valuable class becomes difficult. Fixing this imbalance is crucial for a helpful model. The model will be biased towards 'not fraud' results.


#scaling


robust_scaler = RobustScaler()

features = payment_fraud_df.drop(columns=['isFraud'])
target = payment_fraud_df['isFraud']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=1)

robust_scaler.fit(X_train)
robust_scaled_X_train = robust_scaler.transform(X_train)

robust_scaler.fit(X_test)
robust_scaled_X_test = robust_scaler.transform(X_test)

print("Per feature minimum before scaling")
print(X_train.min(axis=0))

# robust scaling
print("Per feature minimum after robust scaling")
print(robust_scaled_X_train.min(axis=0))

print("Per feature maximum before scaling")
X_train.max(axis=0)

print("Per feature minimum after robust scaling")
print(robust_scaled_X_train.max(axis=0))

X_train
robust_scaled_X_train


#logistic regression
logistic_regression_classifier = LogisticRegression(max_iter=5000)
logistic_regression_classifier.fit(robust_scaled_X_train, Y_train)


Y_predicted_logistic = logistic_regression_classifier.predict(robust_scaled_X_test)


# Logistic Regression confusion matrix and results
cm_logistic = confusion_matrix(Y_test, Y_predicted_logistic)
ConfusionMatrixDisplay(cm_logistic).plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

print("Logistic Regression model results:")
print(f"Accuracy:  {round(accuracy_score(Y_test, Y_predicted_logistic)*100 , 2)}%")
print(f"Precision:  {round(precision_score(Y_test, Y_predicted_logistic)*100 , 2)}%")
print(f"Recall:  {round(recall_score(Y_test, Y_predicted_logistic)*100 , 2)}%")
print(f"F1 Score:  {round(f1_score(Y_test, Y_predicted_logistic)*100 , 2)}%")

# Decision tree 

tree_classifier =  DecisionTreeClassifier()
tree_classifier.fit(X_train, Y_train)

Y_predicted_tree = tree_classifier.predict(X_test) #decision tree does not need scaling

# Decision tree confusion matrix and results
cm_tree = confusion_matrix(Y_test, Y_predicted_tree)
ConfusionMatrixDisplay(cm_tree).plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("Decision tree model results:")
print(f"Accuracy:  {round(accuracy_score(Y_test, Y_predicted_tree)*100 , 2)}%")
print(f"Precision:  {round(precision_score(Y_test, Y_predicted_tree)*100 , 2)}%")
print(f"Recall:  {round(recall_score(Y_test, Y_predicted_tree)*100 , 2)}%")
print(f"F1 Score:  {round(f1_score(Y_test, Y_predicted_tree)*100 , 2)}%")


# Neural Networks

#scaled

NN_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
NN_model.fit(robust_scaled_X_train, Y_train)

Y_predicted_NN = NN_model.predict(robust_scaled_X_test)

print(f"Robust scaled Neural Network Accuracy: {round(accuracy_score(Y_test, Y_predicted_NN)*100, 2)}%")


# Neural network confusion matric and results
cm_NN = confusion_matrix(Y_test, Y_predicted_NN)
ConfusionMatrixDisplay(cm_NN).plot(cmap="Blues")
plt.title("Neural Netword Confusion Matrix")
plt.show()

print(f"Neural Netword results")
print(f"Accuracy:  {round(accuracy_score(Y_test, Y_predicted_NN)*100 , 2)}%")
print(f"Precision:  {round(precision_score(Y_test, Y_predicted_NN)*100 , 2)}%")
print(f"Recall:  {round(recall_score(Y_test, Y_predicted_NN)*100 , 2)}%")
print(f"F1 Score:  {round(f1_score(Y_test, Y_predicted_NN)*100 , 2)}%")



# ROC, AUC
supervised_models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Neural Network": MLPClassifier()
}


for model_name, model_classifier in supervised_models.items():
    model_classifier.fit(robust_scaled_X_train, Y_train)

    Y_probability_predict = model_classifier.predict_proba(robust_scaled_X_test)[:, 1]

    false_positive_rate, true_positive_rate, _ = roc_curve(Y_test, Y_probability_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC = {round(roc_auc, 2)})")


plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Titanic Dataset")
plt.legend(loc="lower right")
plt.grid()
plt.show()

models = ['Logistic Regression', 'Decision Tree', 'Neural Network']
accuracies = [99.93, 99.94, 99.95]

bars = plt.bar(models, accuracies)
plt.ylim(99.90, 100.00)
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy of Classification Models')

# final report 
print("Logistic Regression:")
print(classification_report(Y_test, Y_predicted_logistic))

print("Decision Tree")
print(classification_report(Y_test, Y_predicted_tree))

print("Neural Network")
print(classification_report(Y_test, Y_predicted_NN))