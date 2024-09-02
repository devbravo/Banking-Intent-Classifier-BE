import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import os


nltk.data.path.append('nltk_data')
nltk.data.load(os.path.join('nltk_data', 'tokenizers/punkt/PY3/english.pickle'))
nltk.data.load(os.path.join('nltk_data', 'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle'))
nltk.data.load(os.path.join('nltk_data', 'corpora/stopwords/english'), format='text')

def clean_text(text):
    """
    Descr: Clean text data by removing punctuation, stopwords, and converting to lowercase.
    Input: text
    Output: cleaned text
    """
    tokens = re.sub(r"\{\{.*?\}\}", "", text)
    tokens = word_tokenize(tokens)
    tokens = [w.lower() for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
  
def nltk_to_wordnet_pos(nltk_tag):
  """
  Descr: Convert NLTK POS tag to WordNet POS tag.
  Input: nltk_tag
  Output: wordnet_pos
  """
  if nltk_tag.startswith('J'):
      return wordnet.ADJ
  elif nltk_tag.startswith('V'):
      return wordnet.VERB
  elif nltk_tag.startswith('N'):
      return wordnet.NOUN
  elif nltk_tag.startswith('R'):
      return wordnet.ADV
  else:
      return None

def lemmatizer(data):
  """
  Descr: Lemmatize text data using WordNetLemmatizer.
  Input: data
  Output: lemmatized_words
  """
  lemmatizer = WordNetLemmatizer()
  tokens = word_tokenize(data)
  pos_tags = pos_tag(tokens)

  lemmatized_words = []
  for word, tag in pos_tags:
      wn_tag = nltk_to_wordnet_pos(tag)
      if wn_tag is None:
          lemmatized_word = lemmatizer.lemmatize(word)
      else:
          lemmatized_word = lemmatizer.lemmatize(word, pos=wn_tag)
      lemmatized_words.append(lemmatized_word)

  return lemmatized_words

def numericalize(vocab, data):
    """
    Descr: Numericalize a list of documents or a single document.
    Input: data - a list of documents or a single document
           vocab - a dictionary mapping tokens to indices
    Output: indexed_data - a list of indexed documents or a single indexed document
    """
    indexed_data = []
    indexed_seq = [vocab.get(token, vocab['<UNK>']) for token in data]
    indexed_data.append(indexed_seq)

    return indexed_data