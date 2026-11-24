import nltk

nltk.download('stopwords')
 
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

print(list(stop_words)[:10])