import os
import nltk
#nltk.download('stopwords')
#nltk.download('popular')

# load BERT's summarizer, which we'll use to make summaries of our text
from summarizer import Summarizer
import re
import pke
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import requests
import json
import random
import pywsd
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn

# load summarizer model
model = Summarizer()
