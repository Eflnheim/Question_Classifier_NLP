# Import Library
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import  SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Membaca Dataset
train_data = open('training_data.txt', 'r+')
test_data = open('test_dataset.txt', 'r+')

train = pd.DataFrame(train_data.readlines(), columns = ['Question'])
test = pd.DataFrame(test_data.readlines(), columns = ['Question'])

train.head()

# Memisahkan pertanyaan dan kategori coarse serta fine
train['QType'] = train.Question.apply(lambda x: x.split(' ', 1)[0])
train['Question'] = train.Question.apply(lambda x: x.split(' ', 1)[1])
train['QType-Coarse'] = train.QType.apply(lambda x: x.split(':')[0])
train['QType-Fine'] = train.QType.apply(lambda x: x.split(':')[1])
test['QType'] = test.Question.apply(lambda x: x.split(' ', 1)[0])
test['Question'] = test.Question.apply(lambda x: x.split(' ', 1)[1])
test['QType-Coarse'] = test.QType.apply(lambda x: x.split(':')[0])
test['QType-Fine'] = test.QType.apply(lambda x: x.split(':')[1])

# Menghapus QType dan QType-Fine karena hanya fokus pada klasifikasi kategori coarse
train.pop('QType')
train.pop('QType-Fine')
test.pop('QType')
test.pop('QType-Fine')

# Menampilkan daftar kelas (kategori)
classes = np.unique(np.array(train['QType-Coarse']))
classes

# Melakukan Label Encoding untuk mengubah kelas menjadi representasi angka
le = LabelEncoder()
le.fit(pd.Series(train['QType-Coarse'].tolist() + test['QType-Coarse'].tolist()).values)
train['QType-Coarse'] = le.transform(train['QType-Coarse'].values)
test['QType-Coarse'] = le.transform(test['QType-Coarse'].values)

# Melakukan pra-pemrosesan pada dataset
all_corpus = pd.Series(train.Question.tolist() + test.Question.tolist()).astype(str)

def text_clean(corpus):
  '''
  Purpose : Function to keep only alphabets, digits, and certain words (punctuation, qmarks, tabs etc. removed)

  Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained even after the cleaning process

  Output : Returns the cleaned text corpus
  '''
  cleaned_corpus = [] # Initialize an empty list
  for row in corpus:
    qs = []
    for word in row.split():
      p1 = re.sub(pattern='[^a-zA-Z]', repl=' ', string=word)
      p1 = p1.lower()
      qs.append(p1)
    cleaned_corpus.append(' '.join(qs)) # Append the cleaned row to the list

  return pd.Series(cleaned_corpus) # Convert list  to pandas Series and return it

def stopwords_removal(corpus):
  wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
  stop = set(stopwords.words('english'))
  for word in wh_words:
    stop.remove(word)
  corpus = [[x for x in x.split() if x not in stop] for x in corpus]
  return corpus

def lemmatize(corpus):
  lem = WordNetLemmatizer()
  corpus = [[lem.lemmatize(x, pos='v') for x in x] for x in corpus]
  return corpus

def stem(corpus, stem_type=None):
  if stem_type == 'snowball':
    stemmer = SnowballStemmer(language='english')
    corpus = [[stemmer.stem(x) for x in x] for x in corpus]
  else:
    stemmer = PorterStemmer()
    corpus = [[stemmer.stem(x) for x in x] for x in corpus]
  return corpus

def preprocess(corpus, cleaning=True, stemming=False, stem_type=None, lemmatization=False, remove_stopwords=True):
  '''
  Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)

  Input :
  'corpus' - Text corpus on which pre-processing tasks will be performed

  'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should
                                                                be performed or not
  'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter Stemmer. 'snowball' corresponds to Snowball Stemmer

  Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together

  Output : Returns the processed text corpus
  '''
  if cleaning == True:
    corpus = text_clean(corpus)

  if remove_stopwords == True:
    corpus = stopwords_removal(corpus)
  else:
    corpus = [[x for x in x.split()] for x in corpus]

  if lemmatization == True:
    corpus = lemmatize(corpus)

  if stemming == True:
    corpus = stem(corpus, stem_type)

  corpus = [' '.join(x) for x in corpus] # Join the words back into a sentence
  return corpus

all_corpus = preprocess(all_corpus, remove_stopwords=True)

train_corpus = all_corpus[0:train.shape[0]]
test_corpus = all_corpus[train.shape[0]:]

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
tf_idf_matrix_train = vectorizer.fit_transform(train_corpus)
tf_idf_matrix_test = vectorizer.transform(test_corpus)

# Menggunakan Keras untuk membangun arsitektur model jaringan saraf
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Mengonversi label kelas menjadi vektor one-hot encoding
y_train = to_categorical(train['QType-Coarse'], train['QType-Coarse'].nunique())
y_test = to_categorical(test['QType-Coarse'], train['QType-Coarse'].nunique())

# Mendefinisikan dan membangun arsitektur jaringan 
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=tf_idf_matrix_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# Pelatihan Model
training_history = model.fit(tf_idf_matrix_train, y_train, epochs=10, batch_size=100)

# Evaluasi model pada data uji berdasarkan akurasi
loss, accuracy = model.evaluate(tf_idf_matrix_test, y_test, verbose=False)
print('Testing Accuracy: {:.4f}'.format(accuracy))

import h5py
# Save the model architecture to JSON
model_structure = model.to_json()
with open('question_classification_model.json', 'w') as json_file:
  json_file.write(model_structure)

# Save the model weights (make sure the filename ends in '.weights.h5')
model.save_weights('question_classification_weights.weights.h5')

import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)