import numpy as np
from datasets import load_dataset
from collections import defaultdict
import os

dataset = load_dataset("Deysi/spam-detection-dataset")
n_emails=0
n_spam=0
n_non_spam=0
emails=[]
for item in dataset['train']:  
    text = item['text']  
    label = item['label']  
    if label=='spam':
        mail=[text,1]
        emails.append(mail)
        n_spam+=1
    else:
        mail=[text,0]
        emails.append(mail)
        n_non_spam+=1

    n_emails+=1
    
# print(n_emails)
# print(emails[8174])
p_spam=float(n_spam/n_emails)
p_non_spam=float(n_non_spam/n_emails)

# Function to create vocabulary
def create_vocabulary(emails):
    vocabulary = defaultdict(int)
    spam_vocab=defaultdict(int)
    non_spam_vocab=defaultdict(int)
    for email, label in emails:
        words = set(email.lower().split())
        for word in words:
            vocabulary[word] += 1
            if label==1:
                spam_vocab[word]+=1
            else:
                non_spam_vocab[word]+=1
    return vocabulary,spam_vocab,non_spam_vocab

# Create vocabulary
vocabulary,spam_vocab,non_spam_vocab = create_vocabulary(emails)

def calculate_word_probabilities(vocabulary, spam_vocab, non_spam_vocab):
    word_probabilities = defaultdict(lambda: [0, 0])  # [P(word|spam), P(word|non-spam)]
    for word, count in vocabulary.items():
        #Laplace smoothing
        p_word_spam = (spam_vocab[word] + 1) / (n_spam+1)
        p_word_non_spam = (non_spam_vocab[word] + 1) / (n_non_spam+1)
        word_probabilities[word] = [p_word_spam, p_word_non_spam]

    return word_probabilities

word_probabilities = calculate_word_probabilities(vocabulary, spam_vocab, non_spam_vocab)

def predict_email_label(email_text, word_probabilities, p_spam, p_non_spam):
    words = email_text.split()
    log_prob_spam = np.log(p_spam)
    log_prob_non_spam = np.log(p_non_spam)

    for word in words:
        if word in word_probabilities:
            log_prob_spam += np.log(word_probabilities[word][0])
            log_prob_non_spam += np.log(word_probabilities[word][1])
    # Choose the class with higher probability
    return int(log_prob_spam > log_prob_non_spam)

test_emails = load_dataset("Deysi/spam-detection-dataset", split='test')  # Load test dataset

t_emails=[]
for item in test_emails:  
    text = item['text']  
    label = item['label']  
    if label=='spam':
        mail=[text,1]
        t_emails.append(mail)
    else:
        mail=[text,0]
        t_emails.append(mail)

test_labels = np.array([email[1] for email in t_emails])
prediction=np.zeros(len(t_emails))
for i in range(len(t_emails)):
    prediction[i]=predict_email_label(t_emails[i][0],word_probabilities,p_spam,p_non_spam)
    
accuracy = np.mean(prediction == test_labels)
print("Accuracy for hugging face test data:", accuracy*100)


def read_emails_from_folder(folder_path, vocabulary):
    predictions = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            email_text = file.read().lower()
            prediction = predict_email_label(email_text,word_probabilities,p_spam,p_non_spam)
            print('prediction for',file_name,'is',prediction)
            predictions.append(prediction)
    return predictions

test_folder_path = "test"
test_predictions = read_emails_from_folder(test_folder_path, vocabulary)
print(test_predictions)
