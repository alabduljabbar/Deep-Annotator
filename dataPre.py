from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv
import pickle
import string
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from nltk.tokenize import RegexpTokenizer


labels_file='labels.csv'
policies_file='data.txt'

    #two lists, one for labels, the other for privacy policies
labels = []
policies_ = []
policies=[]
with open(labels_file, 'r') as f:
    labels = [[int(x) for x in label] for label in csv.reader(f, delimiter=',')]
    f.close()

with open(policies_file, 'r') as f:
    policies_= f.read().split('-----------------------------------------------------------------------')
    f.close()

for p in policies_:
	policies.append(" ".join("".join([" " if ch in string.punctuation else ch for ch in p]).split()))

    # Creating pickle file for labels
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
    f.close()

    # Creating pickle file for policies
with open('policies.pkl', 'wb') as f:
    pickle.dump(policies, f)
    f.close()

    # Bag Of Words
vectorizer = CountVectorizer(lowercase=True,stop_words='english',max_features=10000,max_df=.7)
X = vectorizer.fit_transform(policies)
X.toarray()

	# Creating pickle file for Bag Of Words

with open('bagOfWords.pkl', 'wb') as f:
    pickle.dump(X.toarray(), f)
    f.close()

    # ngram from 1 to 4
vectorizer = TfidfVectorizer(lowercase=True,stop_words='english',max_features=10000,ngram_range=(1,4))
X2 = vectorizer.fit_transform(policies)
X2.toarray()
	
	# Creating pickle file for ngram from 1 to 4
with open('ngram1_4.pkl', 'wb') as f:
    pickle.dump(X2.toarray(), f)
    f.close()

    # ngram from 2 to 4
vectorizer = TfidfVectorizer(lowercase=True,stop_words='english',max_df=.7,max_features=10000,ngram_range=(2,4))
X3 = vectorizer.fit_transform(policies)
X3.toarray()

	# Creating pickle file for ngram from 2 to 4
with open('ngram2_4.pkl', 'wb') as f:
    pickle.dump(X3.toarray(), f)
    f.close()
    # Word Embeddings for policies
message_embeddings_list=[]
temp_list=[]
#https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
#https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1
#https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1
#https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
counter=1
for policy in policies:
	print(counter)
	temp_list=[]
	temp_list.append(policy)
	embeddings = embed(temp_list)
	message_embeddings = session.run(embeddings)
	#print(message_embeddings['outputs'])
	message_embeddings = np.array(message_embeddings).tolist()
	message_embeddings_list.append(message_embeddings[0])
	counter+=1
	
	# Creating pickle file for Word Embeddings for policies
with open('wordEmbeddings.pkl', 'wb') as f:
	pickle.dump(message_embeddings_list, f)
	f.close()
