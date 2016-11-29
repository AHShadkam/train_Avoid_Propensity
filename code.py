import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")



nltk.data.path.append('./nltk_data/')



stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['ndcs', 'ndc', 'dcs','NDCS'])
# This is important in bi and trigrams!
removelist = ['not','no','in','out','on','off','under','below','down','too']
[stopwords.remove(element) for element in removelist if element in stopwords]


def _get_lower_case(text):
    return str(text).lower()

def _remove_special_character(text):
    return text.replace('/',' ')

def _get_stopword_filtered_tokens(text):
    return [word for word in nltk.word_tokenize(str(text)) if word not in stopwords]

def _filter_out_punctuation(tokens):
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z0-9]', str(token)):
            filtered_tokens.append(token)
    return filtered_tokens

def _stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def _tokenize_and_stem(text):
    if(text):
        text= _get_lower_case(text)
        text= _remove_special_character(text)
        tokens = _get_stopword_filtered_tokens(text) 
        tokens = _filter_out_punctuation(tokens)      
        stems = _stem_tokens(tokens)
        return(" ".join(stems))
    else:
        return("")
        

## Read and trim data
data = pd.read_csv("symptom_data.csv", header=0, quoting=3)

def _filter_by_coverage(coverage):
    if str(coverage).upper() in set(['IW', 'SP']):
        return True
    else:
        return False

def _filter_by_recalls(symptoms):
    if symptoms and re.search("recall", str(symptoms).lower().strip()):
        return False
    else:
        return True   
        
def _validate_domain(selected_domain):
	if str(selected_domain).lower() in ['washer','dryer','cooking','refrigeration','lawngarden']:
		return True
	else:
		return False

def _filter_by_domain(domain,selected_domain):
    if domain == selected_domain:
        return True
    else:
        return False  
  

data =data[data['coverage'].apply(_filter_by_coverage)]
data =data[data['symptoms'].apply(_filter_by_recalls)]

selected_domain = str(raw_input("Please enter a domain: "))
while not _validate_domain(selected_domain):
    selected_domain =str(raw_input("please enter one of these domains: washer, dryer, refrigeration, cooking, lawngarden"))

selected_domain=str(selected_domain).lower().title()	
data =data[np.vectorize(_filter_by_domain)(data['domain'],selected_domain)]  
data.reset_index(drop=True)


vocab = [_tokenize_and_stem(i) for i in data['symptoms']]
tfidf_vectorizer = TfidfVectorizer(max_features=3000, tokenizer = None, use_idf=True, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(vocab)
# tfidf_vectorizer.vocabulary_

import os

def makemydir(whatever):
    try:
        os.makedirs(whatever)
    except OSError:
        pass


dir_to_dump='Data-{domain}'.format(domain=selected_domain)
makemydir(dir_to_dump)
joblib.dump(tfidf_matrix,'{dir}/tfidf_vectorizer.pkl'.format(dir=dir_to_dump)) 