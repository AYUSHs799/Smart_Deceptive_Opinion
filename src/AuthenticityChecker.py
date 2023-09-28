# Library import
import sys
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os


def clean_text(text):
    """ 
    This function will be used to clean the Corpus
    input: text
    output: cleaned text
    """
    text = text.lower()                                                             # Convert to lowercase
    text = re.sub(r'http\S+', '', text)                                             # Remove URLs
    text = re.sub(r'<.*?>', '', text)                                               # Remove HTML tags
    text = re.sub(r'(!.*)','', text)                                                # Remove gif
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)                                        # Remove special characters
    tokens = word_tokenize(text)                                                    # Tokenization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]                        # Lemmatization
    cleaned_text = ' '.join(tokens)                                                 # Joining the tokens
    return cleaned_text  


if __name__ == "main":

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Check if the path to the corpus directory is given
    if sys.argv[1] == None:
        print("Please provide the path to the corpus directory")
        exit()

    # Check if the comment is given
    if sys.argv[2] == None:
        print("Please provide Comment to test for authenticity")
        exit()

    # Check if the model name is given
    if sys.argv[3] == None:
        print("Please provide Name of the Model to be used. Available Models: Logclg, RFclf")
        exit()
    
    # Check if the model name is valid
    if sys.argv[3] not in ['Logclf', 'RFclf']:
        print("Please provide a valid Model to be used. Available Models: Logclg, RFclf")
        exit()

    corpus_dir = sys.argv[1]
    comment = sys.argv[2]
    clf_type = sys.argv[3]

    # corpus_dir is not given properly
    if not os.path.isdir(corpus_dir):
        print("Directory does not exist")
        exit()

    Data = pd.read_csv('./deceptive-opinion.csv')
    Data['cleaned_text'] = Data['text'].apply(clean_text)

    vectorizer = TfidfVectorizer()                                      # sklearn tf-idf vectorizer
    X = vectorizer.fit_transform(Data['cleaned_text'])                  # Creating vectors

    embed = X.toarray()                                                 # Converting the embeddings to an array
    embed_frame = pd.DataFrame(embed)               
    Features = pd.concat([Data,embed_frame],axis=1)                     # Collecting everything in a Dataframe
    Features['y'] = np.where(Features['deceptive']=='truthful',1,0)     # One Hot Encoding the output 
    Features = Features.drop(columns=['hotel','polarity','source','text','cleaned_text','deceptive'],)
    X,y= Features.iloc[:,:-1],Features.iloc[:,-1]
    comment_text = clean_text(comment)
    comment_vector = vectorizer.transform([comment_text])

    if clf_type == 'RFclf':
        rfclf = RandomForestClassifier(n_estimators=500,)
        rfclf.fit(X,y)
        y_hat = rfclf.predict(comment_vector)

    if clf_type == 'Logclg':
        logclg = LogisticRegression()
        logclg.fit(X,y)
        y_hat = logclg.predict(comment_vector)
    
    if y_hat[0] == 1:
        print("The comment is truthful")
    else:
        print("The comment is deceptive")

    
