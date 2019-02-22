from Truecaser import *
import pickle
import nltk
import string

class TrueCaser:

    def __init__(self, trained_fname):
        self._load(trained_fname)
        
    def _load(self, fname):
        print("loading {} in truecaser...".format(fname))
        f = open(fname, 'rb')
        self.uniDist = pickle.load(f)
        self.backwardBiDist = pickle.load(f)
        self.forwardBiDist = pickle.load(f)
        self.trigramDist = pickle.load(f)
        self.wordCasingLookup = pickle.load(f)
        f.close()
        
    def truecase(self, tokens):
        tokensTrueCase = getTrueCase(tokens, 'title', self.wordCasingLookup, self.uniDist, self.backwardBiDist, self.forwardBiDist, self.trigramDist)
        return tokensTrueCase

    def truecase_fast(self, tokens):
        out = []
        for i, t in enumerate(tokens):
            if i == 0:
                out.append(t.title())
            else:
                if t.lower() in self.wordCasingLookup:
                    opts = list(self.wordCasingLookup[t.lower()])
                    cnts = [self.uniDist[o] for o in opts]
                    ind = cnts.index(max(cnts))
                    out.append(opts[ind])
                else:
                    out.append(t.title())
        return out


if __name__ == "__main__":       
    t = TrueCaser('english_distributions.obj')

    print(t.truecase("john loves mary".split()))
    print(t.truecase_fast("john loves mary".split()))
