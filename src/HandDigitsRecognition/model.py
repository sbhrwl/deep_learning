# from IPython.display import Image  # to display images
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_digits  # mnist data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
# %matplotlib inline

digits = load_digits()

print('Stage 1: Data Engineering')
data = digits.images
target = digits.target

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data[i], cmap='gray')
    plt.xlabel(target[i])
plt.show()

print("Data shape is: ", data.shape)
print("Target shape is: ", target.shape)

print('Stage 2: Feature Engineering')
# reshaping the input before passing the input to MinMaxScaler
data = data.reshape((1797, 64, ))
min_max_sc = MinMaxScaler()
X = min_max_sc.fit_transform(data)

print('Stage 3: Split data into training and test set')
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

print('Stage 4: Model Training- Logistic Regression')
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_prediction = lg.predict(X_test)

print('Stage 5: Model Evaluation')
print("Classification Report")
print(classification_report(y_test, y_prediction))
print("Accuracy Score")
print(accuracy_score(y_test, y_prediction))
print("Confusion Matrix")
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_prediction), annot=True)
pd.DataFrame({'Actual': y_test, 'Prediction': y_prediction}).head(50)
