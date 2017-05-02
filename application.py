import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from aliasgenerator import *
from source import *


def main:

	# load a cleaned json files of article
	with open('cleaned_article.json', 'r') as f:
	     data = json.load(f)


	data = filter(None, data) # filter out empty string

	for i, article in enumerate (data):
	    data[i] = [s.encode('utf-8') for s in article] # decode unicode str to str

	corpus = data[:300] # test on a corpus of the last 300 articles

	result = alias_MCMC_lda (corpus,10, 25)

	print (result)
	
if __name__ == "__main__":
    main()  	