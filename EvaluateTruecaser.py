from Truecaser import *
from mytruecaser import TrueCaser
import pickle
import nltk
import string
import timeit

def evaluateTrueCaser(testSentences):
    correctTokens = 0
    totalTokens = 0

    tc = TrueCaser("english_distributions.obj")

    elapsed = 0
    
    for sentence in testSentences:
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        start = timeit.default_timer()
        tokensTrueCase = tc.truecase(tokens)
        stop = timeit.default_timer()
        print('Time: ', stop - start)  

        elapsed += stop - start
        
        perfectMatch = True
        
        for idx in range(len(tokensCorrect)):
            totalTokens += 1
            if tokensCorrect[idx] == tokensTrueCase[idx]:
                correctTokens += 1
            else:
                perfectMatch = False
        
        if not perfectMatch:
            print(tokensCorrect)
            print(tokensTrueCase)
        
            print("-------------------")
    
    print("Avg time {}".format(elapsed / len(testSentences)))
    print("Accuracy: %.2f%%" % (correctTokens / float(totalTokens)*100))
    
    
def defaultTruecaserEvaluation():
    fname = "testsentences.txt"
    with open(fname) as f:
        testSentences = f.read().split("\n")
    
    evaluateTrueCaser(testSentences)
    
if __name__ == "__main__":
    
    defaultTruecaserEvaluation()

