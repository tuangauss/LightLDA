import random
import numpy as np


def GenerateAlias (p):
	# p is a list of probability
    threshold = 1/float(len(p))
    indices = range(1, len(p) + 1)
    # initialize integer sets
    Greater = dict((i,float(p[i-1])) for i in indices if p[i-1] >= threshold)
    Smaller = dict((i,float (p[i-1])) for i in indices if p[i-1] < threshold)
    
    table = []
    
    while Greater and Smaller :
        k = random.choice(Greater.keys())
        l = random.choice (Smaller.keys())
        table.append([k,l,Smaller[l]]) # store the donor, receiver and probability
        
        Greater[k] = float(Greater[k] - (threshold - Smaller[l])) # As k gives to l, reduce probability qk
        del Smaller[l] # probability of l is finalized
        
        if Greater[k] < threshold:
             # if k gives too much, move to Smaller
            Smaller.update({k:Greater[k]})
            del Greater[k]
            
    if Smaller: 
        table.append([Smaller.keys()[0], None, Smaller[Smaller.keys()[0]]])
    else:
        table.append([Greater.keys()[0], None, Greater[Greater.keys()[0]]])
    return table


def SampleAlias (A,l):
    index = random.randint(0,l-1)
    if A[index][1]:
        if l*A[index][2] > np.random.uniform(0,1,1):
            return A[index][1]
        else:
            return A[index][0]
    else:
        return A[index][0]
    