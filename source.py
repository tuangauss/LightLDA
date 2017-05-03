import random
import numpy as np
import pandas as pd
from collections import Counter
from aliasgenerator import *


def random_topic (corpus, no_topic): # randomly assign each word to one of the topic
    result = [np.random.randint(low=1, high= no_topic+1, size = len(corpus[i]), dtype = 'int').tolist() for i in range(len(corpus))]
    return result

def word_in_topic (flatten_data, flatten_topic, no_topic, word): # how many times a word has been assigned to a particular topic
    indices = [h for h, x in enumerate(flatten_data) if x == word]
    occurences = dict(Counter(np.array(flatten_topic)[indices]))
    #### make sure that we cover all topic (topic that does not get assign to 'word' get 0 value)
    for h in range (1, no_topic+1):
        if h not in occurences.keys():
            occurences[h] = 0
    return occurences.values()

def topic_in_text (text,no_topic): # distribution of the words assigned to a topic in a text
    occurences = dict(Counter(text))
    #### make sure that we cover all topic (topic that does not get assign to 'word' get 0 value)
    for h in range (1, no_topic+1):
        if h not in occurences.keys():
            occurences[h] = 0
    return occurences.values()

def word_dist_per_topic(word_topic_corpus, topic_in_corpus, topic_index):
    # for each topic, find ratio that each word carries that topic
    list_of_words = word_topic_corpus.keys()
    word_dist = [float(word_topic_corpus[word][topic_index])/topic_in_corpus[topic_index] if topic_in_corpus[topic_index] != 0 else 0
                 for word in list_of_words]
        # get top 10
    word_dist_rank_index = np.argsort(word_dist).tolist()[-10:] # since they sort by increasing order
    word = [word_topic_corpus.keys()[i] for i in word_dist_rank_index]
    percent = [word_dist[j] for j in word_dist_rank_index]
    
    dictionary = dict(zip(word, percent))
    return (dictionary)

def normalize (prob_array): #normalized an array of probability
    sum = np.sum(prob_array)
    result = list(np.array(prob_array).astype(np.float)/sum)
    return result


def alias_MCMC_lda (corpus, no_topic, no_interative = 25, traces = True, alpha = 1, beta = 1):
	# default number of iterative is 25
	if alpha < 1:
		print ("Please initialize alpha greater than or equal to 1")
		break # break the loop if alpha < 1
	if beta < 1:
		print ("Please initialize beta greater than or equal to 1")
		break # break the loop if beta is < 1
    if traces:
        print ("Initializing data, please wait")
    flatten_data =  [item for f in range(len(corpus)) for item in corpus[f]] # flatten all articles in corpus into words
    topic_matrix = random_topic(corpus,no_topic)  # assign random topic to all words in the corpus
    unique_word = list(set(flatten_data))
    flatten_topic = [item for f in range(len(topic_matrix)) for item in topic_matrix[f]]
    
    # initialize logging matrices
    # how many times a word has been assigned a particular topic
    word_topic_corpus = [word_in_topic(flatten_data, flatten_topic, no_topic, word) for word in unique_word]
    word_topic_corpus = dict(zip(unique_word, word_topic_corpus))

    topic_in_document= [topic_in_text(text, no_topic) for text in topic_matrix] # how many times a topic has been assigned in a text
    topic_in_corpus = topic_in_text(flatten_topic, no_topic) # how many times a topic has been assigned in the entire text
    


    initial_topic_prob = [float(i)/sum(topic_in_corpus) for i in topic_in_corpus]
    # print (initial_topic_prob)
    # initialize AliasTable
    AliasTable = GenerateAlias(initial_topic_prob)
    
    if traces:
        print ("Done initializing data. Loading 1st iteration")
        
    for k in range(no_interative):
        statement = ["Done ", " iterations. Currently load ", "th interation"]
        for i, text in enumerate (corpus): 
            for j, word in enumerate (text): # loop through all the words of all the articles in the corpus
                
                
                current_topic = topic_matrix[i][j] # current topic
                coinflip = random.randint(0,1) # generate random integer between 0 or 1
                
                if coinflip == 0:
                    topic_proposal_index = random.randint (0, len(text)-1)
                    topic_proposal = topic_matrix[i][topic_proposal_index] # propose a new topic
                    # let beta be 1 => prevent from having zero denominator
                    mh_acceptance = min(1, (float (word_topic_corpus[word][topic_proposal-1] + beta) 
                                        * (topic_in_corpus[current_topic-1] + beta)
                                        /((word_topic_corpus[word][current_topic-1]+beta) * (topic_in_corpus[topic_proposal-1]+beta))))
                else:
                    topic_proposal = SampleAlias(AliasTable, no_topic)
                    mh_acceptance = min(1, (float(topic_in_document[i][topic_proposal-1]+ alpha)/(topic_in_document[i][current_topic-1]+ alpha)))
                
                mh_sample = random.uniform(0,1)
                if (mh_sample >= mh_acceptance): # then accept the proposal, otherwise, reject proposal
                    new_topic_index = topic_proposal - 1
                    
                    # decrement count of old matrices
                    topic_in_document[i][current_topic-1] = topic_in_document[i][current_topic-1]-1
                    topic_in_corpus[current_topic-1] = topic_in_corpus[current_topic-1]-1
                    word_topic_corpus[word][current_topic-1] = word_topic_corpus[word][current_topic-1]-1

                    # increment count of matrices, now that we have updated
                    topic_matrix[i][j] = new_topic_index + 1
                    topic_in_document[i][new_topic_index] = topic_in_document[i][new_topic_index] +1
                    topic_in_corpus[new_topic_index] = topic_in_corpus[new_topic_index] +1
                    word_topic_corpus[word][new_topic_index] = word_topic_corpus[word][new_topic_index] +1
        if k == no_interative-1:
            message = statement[0] + str (no_interative) + " iterations. Wrapping up"
        else:
            message = statement[0] + str(k+1) +  statement[1] + str(k+2) + statement[2]
        if traces:
            print (message)
                                      
    result_text = [normalize(text) for text in topic_in_document]
    summary = {"Number of Topic": no_topic, "Number of iteration": no_interative, 
               "Numer of articles": len(corpus),"Number of unique words": len(unique_word)}
   
    result_topic = [word_dist_per_topic(word_topic_corpus, topic_in_corpus, topic_index) for topic_index in range(no_topic)]
    result = [summary, result_text, result_topic]
    
    return (result) 


def plot_topic (LDA_object):
    for i, topic in enumerate (LDA_object[2]):
        words = topic.keys()
        value = np.array(topic.values())*100
        y_pos = np.arange(len(words))
        
        plt.barh(y_pos, value, align='center', alpha=0.5)
        plt.yticks(y_pos, words)
        plt.xlabel('percentage')
        plt.ylabel('words')
        plt.title("Top word's percentage in topic " + str(i + 1))

        plt.show()

def plot_article (LDA_object, article):
    if (isinstance (article, int) and (article > 0)):
        topic = [("topic " + str(i+1)) for i in range(len(LDA_object[1][article-1]))]
        value = np.array(LDA_object[1][article-1]) *100
        y_pos = np.arange(len(topic))
        
        plt.barh(y_pos, value, align='center', alpha=0.5)
        plt.yticks(y_pos, topic)
        plt.xlabel('percentage')
        plt.ylabel('topic')
        plt.title("Topic distribution in article " + str(article))

        plt.show()

