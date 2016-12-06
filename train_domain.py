import pandas as pd
import numpy as np
import nltk
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import preprocessing


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")



nltk.data.path.append('./nltk_data/')



stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['ndcs', 'ndc', 'dcs'])
# This is important in bi and trigrams!
removelist = ['not','no','in','out','on','off','under','below','down','too']
[stopwords.remove(element) for element in removelist if element in stopwords]

def _get_lower_case(text):
    return str(text).lower()

def _remove_special_character(text):
	#remove forward slash,it is common with ndcs/
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
        

def create_tfidf_vectorizer(data,variable):
    df_symptoms = data[variable]     
    vocab = [_tokenize_and_stem(i) for i in df_symptoms]
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, tokenizer = None, use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(vocab).toarray()
    return tfidf_matrix,tfidf_vectorizer
	

def makemydir(whatever):
    try:
        os.makedirs(whatever)
    except OSError:
        pass

def _clean_model(x):
    return( x[0:3].lower() if x and len(x) > 2 else "   " )

def ageBucket(x):
    if x <= 6:
        return("0")
    elif x > 6 and x <= 18:
        return "1"
    elif x > 18 and x <= 42:
        return "2"
    elif x > 42 and x <= 91:
        return "3"
    else:
        return "4"

def _make_dummies(data, variables):
    #convert to expected variables only
    filtered_data = data.loc[:,variables].astype(str)
    data_dict = [dict(r.iteritems()) for _, r in filtered_data.iterrows()]
    vectorizer = DV( sparse = False )
    vec_x_cat = vectorizer.fit_transform(data_dict)
    return vec_x_cat, vectorizer

def create_gl_vectorizer(data,variables):
    local_data = data.loc[:,variables]
    local_data['coverage'] = local_data['coverage'].apply(lambda x: x.upper().strip())
    local_data['model'] = local_data['model'].apply(str)
    local_data['model_short'] = local_data['model'].apply(_clean_model)
    local_data['age'] = local_data['age'].apply(int)
    local_data['age_bucket'] = local_data['age'].apply(ageBucket)
    local_data['brand'] = local_data['brand'].apply(lambda x: str(x).lower().strip())
    vec_x_cat, gl_vectorizer = _make_dummies(local_data, variables=['brand', 'model_short', 'coverage', 'age_bucket'])
    #gl_vectorizer.feature_names_
    return vec_x_cat, gl_vectorizer
 
def create_std_scale(data, variable = ['month']):
    local_data = data.loc[:,variable]
    local_data.replace('', np.nan, inplace=True)
    local_data.fillna(0, inplace = True)
    local_data=local_data.astype(float)
    local_data= local_data.as_matrix()
#    std_scale = preprocessing.StandardScaler().fit(local_data.reshape(-1,1))
    std_scale = preprocessing.StandardScaler().fit(local_data)
    X_num = std_scale.transform(local_data)
    return X_num, std_scale
	

# read data file for specific domain
# it should contain the ('symptoms','brand','model', 'coverage', 'age','month')    
data = pd.read_csv("symptom_data.csv", header=0, quoting=3)


tfidf_matrix, tfidf_vectorizer = create_tfidf_vectorizer(data,'symptoms')
vec_x_cat, gl_vectorizer  = create_gl_vectorizer(data,variables=['brand','model', 'coverage', 'age'])
X_num, std_scale = create_std_scale(data)

X_train = np.hstack(( X_num, vec_x_cat, tfidf_matrix ))


# Initialize a Random Forest classifier with 10 trees
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( X_train, data['result'] )




# save the trained algorithms
dir_to_dump='tfidf_vectorizer'
makemydir(dir_to_dump)
joblib.dump(tfidf_vectorizer,'tf_idf_vectorizer/tfidf_vectorizer.pkl') 

dir_to_dump='gl_vectorizer'
makemydir(dir_to_dump)
joblib.dump(vectorizer,'gl_vectorizer/vectorizer_age_buckets.pkl')

dir_to_dump='std_scale'
makemydir(dir_to_dump)
joblib.dump(std_scale,'std_scale/std_scale_age_buckets.pkl')

dir_to_dump='clf'
makemydir(dir_to_dump)
joblib.dump(std_scale,'clf/sgd_age_buckets.pkl')
