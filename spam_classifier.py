import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

df = pd.read_csv('mail_data.csv')
data = df.where((pd.notnull(df)), '')

data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X = data['Message']
Y = data['Category'].astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

print("Training Accuracy:", accuracy_score(Y_train, model.predict(X_train_features)))
print("Test Accuracy:", accuracy_score(Y_test, model.predict(X_test_features)))

if os.path.exists("gmail_emails.csv"):
    gmail_df = pd.read_csv("gmail_emails.csv")
    gmail_df = gmail_df.where((pd.notnull(gmail_df)), '')
    gmail_df['message'] = gmail_df['subject'].fillna('') + ' ' + gmail_df['body'].fillna('')

    gmail_df['Predicted_Label'] = gmail_df['message'].apply(lambda x: model.predict(feature_extraction.transform([x]))[0])
    gmail_df['Label'] = gmail_df['Predicted_Label'].map({0: 'Spam', 1: 'Ham'})

    gmail_df.to_csv("gmail_spam_ham_classified.csv", index=False)
    print(" Gmail emails classified as Spam/Ham and saved to 'gmail_spam_ham_classified.csv'")

    print("\n Classified Gmail Messages:")
    for idx, row in gmail_df.iterrows():
        print(f"Subject: {row['subject']}")
        print(f"Prediction: {row['Label']}\n")

else:
    print(" gmail_emails.csv not found. Make sure you fetched Gmail messages first.")

while True:
    input_mail = input("\nEnter an email message (or type 'exit' to quit):\n")
    
    if input_mail.lower() == 'exit':
        print("Exiting the spam classifier.")
        break
    input_data_features = feature_extraction.transform([input_mail])
    prediction = model.predict(input_data_features)
    
    if prediction[0] == 1:
        print("This is a Ham (Not Spam) email.")
    else:
        print("This is a Spam email.")




