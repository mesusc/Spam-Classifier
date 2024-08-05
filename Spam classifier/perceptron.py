import numpy as np
from datasets import load_dataset
from collections import defaultdict
import os

dataset = load_dataset("Deysi/spam-detection-dataset")
n_emails=0
emails=[]
for item in dataset['train']:  
    text = item['text']  
    label = item['label']  
    if label=='spam':
        mail=[text,-1]
        emails.append(mail)
    else:
        mail=[text,1]
        emails.append(mail)

    n_emails+=1
    
# print(n_emails)
# print(emails[8174])
# Function to create vocabulary
def create_vocabulary(emails):
    vocabulary = defaultdict(int)
    for email, _ in emails:
        words = email.lower().split() #preprocessing(converting into lower case)
        for word in words:
            vocabulary[word] += 1
    return vocabulary

vocabulary = create_vocabulary(emails)

# Function to convert emails to binary data
def convert_to_binary(emails, vocabulary):
    binary_emails = []
    for email, label in emails:
        words_in_email = set(email.lower().split())  # Precompute set of unique words in the email
        binary_email = [1 if word in words_in_email else 0 for word in vocabulary]
        binary_emails.append((binary_email, label))
    return binary_emails
binary_emails = convert_to_binary(emails, vocabulary)
# print(binary_emails[1])
binary_data = np.array([np.array(email[0]) for email in binary_emails])
labels = np.array([email[1] for email in binary_emails])
n=len(binary_data[0])
omega=np.zeros(n)
def perceptron_train(X_train, y_train, epochs=100):
    omega = np.zeros(X_train.shape[1])  
    old_omega=np.ones(X_train.shape[1])
    itr=0
    for _ in range(epochs):
        if np.array_equal(old_omega, omega):
            # print("converged")
            # print(itr)
            break
        itr+=1
        old_omega = omega.copy()
        for x, y in zip(X_train, y_train):
            activation = np.dot(omega, x)
            prediction = 1 if activation >= 0 else -1
            if prediction != y:
                omega += y * x  # Update w
    return omega

omega = perceptron_train(binary_data, labels)

def perceptron_predict(X_test, omega):
    predictions = np.sign(np.dot(X_test, omega))
    for i in range(len(predictions)):
        if predictions[i]==0:
            predictions[i]=1

    return predictions

test_predictions = perceptron_predict(binary_data, omega)

# Calculate accuracy
accuracy = np.mean(test_predictions == labels)
print("Accuracy for hugging face train data:", accuracy*100)

test_emails = load_dataset("Deysi/spam-detection-dataset", split='test')  

t_emails=[]
for item in test_emails:  
    text = item['text']  
    label = item['label']  
    if label=='spam':
        mail=[text,-1]
        t_emails.append(mail)
    else:
        mail=[text,1]
        t_emails.append(mail)

    n_emails+=1

test_emails_binary = convert_to_binary(t_emails, vocabulary)  
test_data = np.array([np.array(email[0]) for email in test_emails_binary])
test_labels = np.array([email[1] for email in test_emails_binary])

test_predictions2 = perceptron_predict(test_data, omega)

# Calculate accuracy
accuracy = np.mean(test_predictions2 == test_labels)
print("Accuracy for hugging face test data:", accuracy*100)



def read_emails_from_folder(folder_path, omega, vocabulary):
    predictions = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            email_text = file.read().lower()
            binary_email = [1 if word in email_text else 0 for word in vocabulary]
            prediction = np.sign(np.dot(binary_email, omega))
            predictions.append(0 if prediction >= 0 else 1)
            tt=0 if prediction >= 0 else 1
            print('predicted label for',file_name,'is',tt)
    return predictions

test_folder_path = "test"
test_predictions = read_emails_from_folder(test_folder_path, omega, vocabulary)
print(test_predictions)
