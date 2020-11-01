import os
import nltk
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

from summarizer import Summarizer


def get_nouns_multipartite(text, num_questions):
    
    """
    
        Determines what nouns are important by using the TopicRank algorithm, 
        which is implemented on a multipartite graph. It builds on top of the TextRank algorithm
        by implementing a graph-based model derived from the PageRank algorithm. 
        
        The following are relevant resources for this algorithm:
            https://www.aclweb.org/anthology/I13-1062.pdf
            https://smirnov-am.github.io/extracting-keyphrases-from-texts-unsupervised-algorithm-topicrank/
            https://github.com/smirnov-am/pytopicrank
        
        In short, the algorithm works as follows:
            1. Use nltk to identify part-of-speech (POS)
            2. Identify longest sequences of adjectives and nouns, and these will constitute our keyphrases
            3. Convert each keyphrase into term frequency vectors using Bag-of-Words (BOW)
            4. Find clusters of keyphrases, using Hierarchical Agglomerative Clustering (HAC)
            5. Use clusters as graph vertices, and sum of distances between each keyphare of topic pairs as edge weight
            6. Apply PageRank to identify most prominent topics
            7. For topN topics extract most significant keyphrases that represent this topic
    
        Args:
            • text: text to analyze
            • num_questions: number of candidates / key topics that we want to extract
    """
    
    output = []
    
    # initialize our multipartite graph keyphrase extraction model
    extractor = pke.unsupervised.MultipartiteRank()
    
    extractor.load_document(input=text)
    
    # get the POS that we're looking for
    pos = {'PROPN', 'ADJ', 'NOUN'}
    
    # get stoplist, words to avoid
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    
    # select the longest sequence of nouns, adjectives, that do not contain punctuation marks or stopwords
    # and let's choose these as our candidates
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    
    # build Multipartite graph and rank candidates using random walk
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    
    # get the n-highest scored candidates, and let's use these as keyphrases
    keyphrases = extractor.get_n_best(n=num_questions)
    
    for key in keyphrases:
        output.append(key[0])
        
    return output
   
def tokenize_sentence(text):
    
    """
    
        Tokenizes our sentence
        
        e.g., "How are you today?" --> ["How", "are", "you", "today?"]
    
    """
    
    # separate our text into sentences
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    
    # strip away spaces at beginning and end
    sentences = [sentence.strip() for sentence in sentences]

    return sentences
    
def get_sentences_for_keyword(keywords, sentences):
    
    """
    
        For each keyword, find the sentence(s) that correspond to that keyword
    
    """
    
    keyword_processor = KeywordProcessor() # use this implementation as fast alternative to keyword matching
    keyword_sentences = {}
    
    # loop through all keywords
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
        
    # loop through each sentence and keyword
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)
            
    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    
    return keyword_sentences

def get_distractors_wordnet(syn, word):
    
    """
    
        Uses WordNet to find words that can be used as distractors for MC questions
    
    """
    
    distractors = []
    
    word = word.lower()
    
    orig_word = word
    
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
        
    # get any hypernyms (words whose meaning includes the meaning of a more specific word)
    # e.g., "animal" is a hypernym of "elephant"
    # and any hyponyms (words that denote a subcategory of a more general class)
    # e.g., "elephant" is a hyponym of "animal"
    
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    
    # find potential words that can be used as hypernyms/hyponyms
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        
        if name == orig_word:
            continue
        name = name.replace("_", " ") # los_angeles -> los angeles
        name = " ".join(w.capitalize() for w in name.split()) # los angeles -> Los Angeles
        if name is not None and name not in distractors:
            distractors.append(name)
            
    return distractors
    
def get_wordsense(sent, word):
    
    """
    
        Get a sentence of the meaning of a word, in context, using (1) Lesk algorithm and (2) max similarity
        Useful for word sense disambiguation tasks (e.g., one word means different things, 
        based on context)
    
        Paper: https://thesai.org/Downloads/Volume11No3/Paper_30-Adapted_Lesk_Algorithm.pdf
        
        The goal here is to see if the word has synonyms (or words close in meaning)
        that we could potentially use as answer choices
        
    """
    
    word = word.lower()
    
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
        
    # get set of synonyms
    synsets = wn.synsets(word, 'n')
    
    if synsets:
        
        # get similarity between possible synsets of all words in 
        # context sentence and possible synsets of ambiguous words, 
        # to determine "context" of the word of interest and what it 
        # "should" mean
        wup = max_similarity(sent, word, "wup", pos = 'n')
        
        # use Lesk algorithm, which will assume that words in the same
        # "neighborhood", or area of text, will tend to share the same topic. 
    
        adapted_lesk_output = adapted_lesk(sent, word, pos = "n")
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        print(f"No synonyms found for the word {word}")
        return None
    
    
def get_distractors_conceptnet(word):
    
    """
    
        Get distractors using ConceptNet, which connects words and
        phrases in a knowledge graph, and uses distance metrics to 
        calculate similarity
        
        Links:
            http://conceptnet.io/
            https://arxiv.org/pdf/1612.03975.pdf
    
    """
    
    word = word.lower()
    original_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    distractor_list = []

    # get url to get ConceptNet graph
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"

    obj = requests.get(url).json()
    
    for edge in obj["edges"]:
        
        link = edge["end"]["term"]
        
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        
        obj2 = requests.get(url2).json()
        
        for edge in obj2["edges"]:
            word2 = edge["start"]["label"]
            
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
                
    return distractor_list
        
        
def get_distractors(keyword_sentence_mapping):
    
    """
    
        For each of our keywords (each of which denote a key "topic" 
        of our text, get distractors that we can use as alternative
        options for MC questions)
        
    """
    # get output dict to use
    key_distractor_list = {}
    
    # loop through our keywords and sentences for each keyword
    for keyword in keyword_sentence_mapping:

        # check to see if we're going to have synonyms to use
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0], keyword)

        # if we have synonyms, use WordNet to get hypernyms/hyponyms
        if wordsense:
            distractors = get_distractors_wordnet(wordsense, keyword)
            # if we can't get any from WordNet, use ConceptNet
            if len(distractors) == 0:
                distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors

        # otherwise, use ConceptNet
        else:
            distractors = get_distractors_conceptnet(keyword)

            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
                
    return key_distractor_list
