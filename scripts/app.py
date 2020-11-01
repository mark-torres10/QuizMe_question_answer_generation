import streamlit as st
import time
import os
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('popular')
from summarizer import Summarizer # make text summaries
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

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

# import from .py files
import true_false_questions
from true_false_questions import clean_text, get_sentences, get_flattened, get_last_portion, get_rightmost_VP_or_NP
from true_false_questions import get_sentence_completions, sort_by_similarity, generate_sentences
import multiple_choice_questions
from multiple_choice_questions import get_nouns_multipartite, tokenize_sentence, get_sentences_for_keyword
from multiple_choice_questions import get_distractors_wordnet, get_wordsense, get_distractors_conceptnet, get_distractors


########################################################

##### Functions for creating True/False questions ######


########################################################

def get_true_false_questions(text, num_questions):

	"""

		Get true/false questions for the specified text
		Args:
			• text: text for which to create questions
			• num_questions: number of questions to create

		Output:
			• question_answers_list: list of questions, where
			each entry is the question + answers for that question

	"""

	# load GPT2 (for generating false sequences) and BERT (for finding sentence similarity of our real sentence against 
	# our fake sentence
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained("gpt2") # we'll use GPT2 to generate sentences
	# load BERT model
	model_BERT = SentenceTransformer('bert-base-nli-mean-tokens') # we'll use BERT to filter sentences based on similarity

	# load necessary NLP tools + parser
	nltk.download("punkt")
	nlp = spacy.load("en")
	benepar.download("benepar_en2")
	benepar_parser = benepar.Parser("benepar_en2")

	# clean + split text
	text = clean_text(text)
	cleaned_text = get_sentences(text)
	cleaned_text = [clean_text(x) for x in cleaned_text]

	# use parser to split sentences, remove last verb phrase or last noun phrase
	sentence_completion_dict = get_sentence_completions(cleaned_text)

	# get false sentences
	probability_true = 0.5 # probability that we'll add a True statement, rather than the False statement
	num_fake_sentences = 3 # number of fake sentences that we'd like to create for each real partial sentence
	answer_choices = " (a) True  (b) False" # define our answer choices
	question_answers_list = [] # list to hold our questions and answers

	for key_sentence in sentence_completion_dict:

		# get our partial sentence
		partial_sentences_list = sentence_completion_dict[key_sentence]

		# start creating false sentences
		false_sentences = []
    
    	# loop through list of partial sentences
    	for sentence in partial_sentence_list:

    		# create our false sentences
			false_sents = generate_sentences(partial_sentence, key_sentence, num_fake_sentences)
			false_sentences.extend(false_sents)

		for idx, false_sent in enumerate(false_sentences):
        	
        	# for each fake option, we now need to decide if we'll use a fake question or a real question

        	# return the actual question
        	if np.random.uniform() <= probability_true:
        		question = f"(ANSWER: True) {key_sentence} : " + answer_choices + "\n" # e.g., "(Answer: True) : 2 + 2 = 4"
        	# return the false sentence
        	else:
        		question = f"(ANSWER: False) {false_sent} : " + answer_choices + "\n" # e.g., "(Answer: False) : 2 + 2 = 5"

        	# add question to question list
        	question_answers_list.append(question)

    # shuffle our questions
    random.shuffle(question_answers_list)

    # get the first "num_questions" values
    return question_answers_list[:num_questions]


########################################################

########## Functions for creating MC questions #########

########################################################

def create_BERT_summarizer_model():

	"""

		Instantiate BERT Summarizer, which will allow us to 
		make summaries on our text

	"""

	model = Summarizer()
	return model

def init_BERT_summarizer_model(text, model):

	"""

		Load our model with the text (and randomly select subset
		of text to summarize, as per BERT's specifications). 

		Then, get the text that we'll actually be summarizing

		Args: 
			• text: text for which to create questions
			• model: BERT summarizer model

	"""

	result = model(text, min_length = 30, max_length = 500, ratio = 0.5)
	summarized_text = ''.join(result)
	return summarized_text

@st.cache
def get_multiple_choice_questions(text, num_questions, num_options = 4):
	
	"""

		Get multiple choice questions for the specified text
		Args:
			• text: text for which to create questions
			• num_questions: number of questions to create
			• num_options: max # of options per MC question

		Output:
			• question_answers_list: list of questions, where
			each entry is the question + answers for that question

	"""

	# create BERT model
	model = create_BERT_summarizer_model()

	# get text for which we will summarize + create questions
	summarized_text = init_BERT_summarizer_model(text, model)

	# get keywords, which will define the main topics for our questions
	keywords = get_nouns_multipartite(summarized_text, num_questions + 10)  # add extra buffer, since some terms won't have synonyms

	# make sure that we're only getting words that were in our BERT 
	# summarized text (since BERT only uses a subset for its summarization)
	filtered_keys = []
	for keyword in keywords:
		if keyword.lower() in summarized_text.lower():
			filtered_keys.append(keyword)

	# get sentences corresponding to our keywords:
	sentences = tokenize_sentence(summarized_text)
	keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)

	# create questions
	key_distractor_list = {}
	key_distractor_list = get_distractors(keyword_sentence_mapping)

	question_num = 1
	question_answers_list = [] # list to hold our questions and answers

	for each in key_distractor_list:

		# get the sentence that we want to ask about
		sentence = keyword_sentence_mapping[each][0]

		pattern = re.compile(each, re.IGNORECASE)

		# add blank for our answer
		output = pattern.sub("___________", sentence)

		# get our question - e.g., "(Answer: Mt. Everest) : The highest mountain in the world is ________ and it is in Asia."
		question = f" (Answer: {each.capitalize()}) : {output}\n" 

		# populate our choices
		choices = [each.capitalize()] + key_distractor_list[each]
		top_n_options = choices[:num_options]
		random.shuffle(top_n_options)
		option_choices = ["a", "b", "c"]
		if num_options >= 4:
			option_choices.append("d")
		if num_options == 5:
			option_choices.append("e")

		for idx, choice in enumerate(top_n_options):

			# get answer choice (e.g., a) Mt. Elbrus )
			answer_choice = f"\t{option_choices[idx]}) {choice}\n"

			# append to question
			question = question + answer_choice

		# take question, append to list of questions and answers
		question_answers_list.append(question)

		# update question counter
		question_num = question_num + 1

	# now, take "num_questions" amount of questions, return that many questions
	return question_answers_list[:num_questions]



def main():
	
	# text/title
	st.title('"QuizMe": Automated Question and Answer Generator for Any Text, Using Machine Learning')
	st.subheader("By: Mark Torres")
	st.header("This app lets you automatically generate True/False and Multiple Choice questions for any passage or piece of text.")

	# create text box to enter data
	st.subheader("Please enter a paragraph or more about any topic (from horses to Australia to macaroni and cheese) and our app will create test questions about that topic!")
	text = st.text_area("Enter your text:")

	# ask how many questions you want
	st.subheader("Please fill out some additional options below:")

	num_questions = st.number_input("How many questions would you like to create? Typically, somewhere around 10 questions gives the best results",
									 min_value = 0, 
									 max_value = 50, 
									 value = 10, 
									 step = 1, 
									 format = "%i")

	# ask if they want True/False, MC, or both
	type_of_question = st.radio("What types of questions do you want? Currently we support True/False and Multiple Choice questions", 
								options = ["Multiple Choice", "True/False", "Both"], index = 0)

	# submit
	if st.button("Create questions!"):
		# add first success message
		st.success("Successfully submitted!")
		# add spinning wheel
		with st.spinner("Sending data to server"):
			time.sleep(5)
		st.success("Data sent to server. Questions are being created as we speak.")
		# add progress bar (will be important while questions are being created)
		progress_bar = st.progress(0)
		# add loading capability (TODO: get this to time the question generation process - use as decorator?)
		for i in range(5):
			time.sleep(2)
			progress_bar.progress((i * 25))
		# get our questions
		if type_of_question == "Multiple Choice":
			question_answer_list = get_multiple_choice_questions(text, num_questions)
		else:
			question_answer_list = get_true_false_questions(text, num_questions)
		# add success at end
		st.success("Questions were successfully created! Check it out below!")
		# print original text
		st.subheader("Your original text:")
		st.markdown(text)
		st.subheader("Here are the questions that our AI program came up with:")
		# print questions
		for question_num, question in enumerate(question_answer_list):
			question_with_num = f"{question_num + 1})" + question
			st.markdown(question_with_num)

	# TODO: add functionality to choose True/False, MC, or both
	st.subheader("The following are some improvements to the app that are underway:")
	st.subheader("Add true question generation ")
	# TODO: allow them to choose how many choices they want per MC question (let's make it 3, 4, or 5)

	# show results (and highlight correct answer)

	
	# click below to learn more
	st.subheader("If you have any feedback or comments, please feel free to either make a pull request at [https]://github.com/mark-torres10/QuizMe_question_answer_generation or send an email to mark.torres[at]aya.yale.edu")

	# if you have any comments for improvements, submit a pull request!

if __name__ == "__main__":
	main()