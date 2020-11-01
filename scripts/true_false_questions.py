import os
import string
from string import punctuation
import torch
import tensorflow as tf
import transformers
import summa
from summa.summarizer import summarize
import benepar # requires Tensorflow, although we'll use torch otherwise
import nltk
from nltk import tokenize
from nltk.tokenize import sent_tokenize
import re
import spacy
import scipy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2") # we'll use GPT2 to generate sentences
# load BERT model
model_BERT = SentenceTransformer('bert-base-nli-mean-tokens') # we'll use BERT to filter sentences based on similarity
nltk.download("punkt")
nlp = spacy.load("en")
#nltk.load("punkt")
benepar.download("benepar_en2")
benepar_parser = benepar.Parser("benepar_en2")

# load summarizer model
#model = Summarizer()


def clean_text(text):
    """
        Wrapper function to perform any text cleaning 
        that we'd want to do
    """
    text = text.strip(punctuation)
    return text


def get_sentences(text, ratio = 1.0):

    """
        Get our sentences to use. 

        Args:
        	• text: text to summarize
        	• ratio: proportion of sentences to use 
        	(more sentences = more possible questions, but longer evaluation time)
        
    """
    
    # sets up our sentence to be summarized
    sentences = summarize(text, ratio = ratio)
    
    # split by sentence, using sent_tokenize method
    sentences_list = tokenize.sent_tokenize(sentences)
    
    # do some regex cleaning
    cleaned_sentences_list = [re.split(r'[:;]+', x)[0] for x in sentences_list]
    
    return cleaned_sentences_list

def get_flattened(tree):
    
    """
        Flattens the tree structure that we'll get from the Berkley
        parser, to allow us to easily work with it
    """
    
    final_sentence_str = None
    if tree is not None:
        sent_str = [" ".join(x.leaves()) for x in list(tree)]
        final_sentence_str = [" ".join(sent_str)][0]
    return final_sentence_str
        
def get_last_portion(main_string, substring):
    
    """
        Here, we get a string of our last verbphrase or
        nounphrase. 
    """
    
    combined_substring = substring.replace(" ", "")
    
    main_string_list = main_string.split()
    
    last_index = len(main_string_list)
    
    for i in range(last_index):
        
        check_string_list = main_string_list[i:]
        
        check_string = "".join(check_string_list)
        
        if check_string == combined_substring:
            return " ".join(main_string_list[:i])
        
    return None


def get_rightmost_VP_or_NP(tree, last_NP = None, last_VP = None):
    
    """
    
        Recursive function, to get the rightmost verb phrase (VP) 
        or noun phrase (NP), which corresponds to the VP or NP 
        that occurs at the end of the sentence
        
    """
    
    # if we don't have more nodes to traverse, we know we've hit the end
    if len(tree.leaves()) == 1:
        return get_flattened(last_NP), get_flattened(last_VP)
    
    # get our last subtree
    last_subtree = tree[-1]
    
    # check if we either have NP or VP:
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
        
    return get_rightmost_VP_or_NP(last_subtree, last_NP, last_VP)
    
def get_sentence_completions(all_sentences):
    
    """
        Returns a dictionary of our sentences as well 
        as the same sentences, just without their terminal
        VP or NP
    """
    
    sentence_completion_dict = {}
    
    # loop through all of our sentences
    for individual_sentence in all_sentences:

        # parse any additional punctuation
        sentence = individual_sentence.strip(r"?:!.,;") 
        
        # get parsed tree
        tree = benepar_parser.parse(sentence)
        
        last_NP, last_VP = get_rightmost_VP_or_NP(tree)
        
        phrases = []
        
        if last_VP is not None:
            VP_string = get_last_portion(sentence, last_VP)
            if VP_string is not None:
                phrases.append(VP_string)
            else:
                phrases.append("")
        if last_NP is not None:
            NP_string = get_last_portion(sentence, last_NP)
            if NP_string is not None:
                phrases.append(NP_string)
            else:
                phrases.append("")
             
        # get our sentence that we want GPT2 to complete
        longest_phrase = sorted(phrases, key=len, reverse=True)
        
        if len(longest_phrase) == 2:
            first_sentence_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            
            if (first_sentence_len - second_sentence_len) > 4:
                del longest_phrase[1]
                
        if len(longest_phrase) > 0:
            sentence_completion_dict[sentence] = longest_phrase
            
    return sentence_completion_dict
            

def sort_by_similarity(original_sentence, new_sentences_list, num_vals = 3):
    
    """
    
        Sort our GPT-2 generated sentences by how similar they are to our original sentence. 
        We want to select sentences that are not similar to our original sentence (since these
        are going to be the statements that are most clearly false)
        
        We will use BERT to perform the similarity calculation
        
        Args:
            • original_sentence: our original sentence
            • new_sentences_list: our new fake sentences
            • num_vals: number of dissimilar sentences that we want to use
        
    """
    
    # encode the sentences from GPT2 into BERT's format (each sentence is a 1-D vector with 768 columns)
    sentence_embeddings = model_BERT.encode(new_sentences_list)
    
    # do same for original sentence
    original_sentence_list = [original_sentence]
    original_sentence_embeddings = model_BERT.encode(original_sentence_list)
    
    # get number of matches, then loop through and sort by dissimilarity
    number_top_matches = len(new_sentences_list)
    
    dissimilar_sentences = []
    
    for query, query_embedding in zip(original_sentence_list, original_sentence_embeddings):

        # calculate distance between original sentence and false sentences
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        
        # get list of distances + indices, then sort
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x:x[1])
        
        # get dissimilarity score (our distance so far shows)
        # how close they are, so we want embeddings
        # that are far away from the sentence embedding
        for idx, distance in reversed(results[0:number_top_matches]):
            
            score = 1 - distance
            
            if score < 0.9: # arbitrary threshold
                dissimilar_sentences.append(new_sentences_list[idx].strip())
                
    # sort the dissimilar sentences ascending, and get first n (so, lowest scores = furthest away)
    sorted_dissimilar_sentences = sorted(dissimilar_sentences, key = len)
    
    return sorted_dissimilar_sentences[:num_vals]
           
        
def generate_sentences(partial_sentence, full_sentence, num_vals):
    
    """
        Generate false sentences, using GPT2, based off partial sentence

        Args:
        	• partial_sentence: our partial sentence, to be filled in by GPT2
        	• full_sentence: the actual sentence
        	• num_vals: how many fake sentences we'd like to generate
    """
    
    input_ids = torch.tensor([tokenizer.encode(partial_sentence)])
    
    maximum_length = len(partial_sentence.split()) + 80
    
    # get outputs
    sample_outputs = model.generate(input_ids, 
                                    do_sample = True,
                                    max_length = maximum_length, 
                                    top_p = 0.9, 
                                    top_k = 50, 
                                    repitition_penalty = 10.0,
                                    num_return_sequences = 10)
    generated_sentences = []
    
    for i, sample_output in enumerate(sample_outputs):
        
        decoded_sentences = tokenizer.decode(sample_output, skip_special_tokens = True)
        decoded_sentences_list = tokenize.sent_tokenize(decoded_sentences)
        generated_sentences.append(decoded_sentences_list[0])
        
    top_n_sentences = sort_by_similarity(full_sentence, generated_sentences, num_vals)
    
    return top_n_sentences
    
    
    
    










