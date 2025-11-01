import nltk
import streamlit as st 
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
def transform_text(text):
    text =text.lower()
    text = nltk.word_tokenize(text)
    wl= WordNetLemmatizer()
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        y.append(wl.lemmatize(i))
    return " ".join(y)        


with open ('vectorize.pkl','rb') as file: 
    tfidf = pickle.load(file)

with open ('model.pkl','rb') as file: 
    model = pickle.load(file) 


st.title("Spam Classifier")
input=st.text_input("Enter the text")

if st.button('predict'):
  transformed_msg = transform_text(input)

  vector = tfidf.transform([transformed_msg])

  result = model.predict(vector)[0]

  if result ==1:
      st.header('Spam')
  else:
      st.header('Not a spam')    