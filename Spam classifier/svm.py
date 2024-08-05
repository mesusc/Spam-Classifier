import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import os

dataset = load_dataset("Deysi/spam-detection-dataset")
X_train = [item['text'] for item in dataset['train']]
y_train = [1 if item['label'] == 'spam' else 0 for item in dataset['train']]  # Convert 'spam' to 1, 'ham' to 0

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

X_test = [item['text'] for item in dataset['test']]
y_test = [1 if item['label'] == 'spam' else 0 for item in dataset['test']]  # Convert 'spam' to 1, 'ham' to 0
X_test_tfidf = vectorizer.transform(X_test)

C_values = [0.1, 1.0, 10.0, 100.0]
accuracies_linear = []
accuracies_rbf = []

for C in C_values:
    
    svm_classifier_linear = SVC(kernel='linear', C=C)
    svm_classifier_linear.fit(X_train_tfidf, y_train)
    y_pred_linear = svm_classifier_linear.predict(X_test_tfidf)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    accuracies_linear.append(accuracy_linear)
    print("C =", C, "Accuracy (Linear Kernel):", accuracy_linear*100)

 
    svm_classifier_rbf = SVC(kernel='rbf', C=C)
    svm_classifier_rbf.fit(X_train_tfidf, y_train)
    y_pred_rbf = svm_classifier_rbf.predict(X_test_tfidf)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    accuracies_rbf.append(accuracy_rbf)
    print("C =", C, "Accuracy (RBF Kernel):", accuracy_rbf*100)


plt.plot(C_values, accuracies_linear, marker='o', label='Linear Kernel')
plt.plot(C_values, accuracies_rbf, marker='o', label='RBF Kernel')
plt.title('Regularization Parameter vs. Accuracy')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.xscale('log')  
plt.grid(True)
plt.legend()
plt.show()

svm_classifier_linear = SVC(kernel='linear', C=1)
svm_classifier_linear.fit(X_train_tfidf, y_train)
def predict_emails_from_folder(folder_path, svm_classifier, vectorizer):
    predictions = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            email_text = file.read().lower()
            email_tfidf = vectorizer.transform([email_text])
            prediction = svm_classifier.predict(email_tfidf)
            print('predicted label for',file_name,'is',prediction[0])
            predictions.append(prediction[0])  
    return predictions

predictions = predict_emails_from_folder("test", svm_classifier_linear, vectorizer)
print(predictions)