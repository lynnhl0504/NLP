#!/usr/bin/env python
import argparse # optparse is deprecated
import itertools
import nltk
from itertools import islice # slicing for iterators
from textblob import Word
from textblob.wordnet import Synset
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import sys
import re, math
from collections import Counter
import datetime
import gensim
from nltk.tokenize import word_tokenize
from nltk import metrics, stem, tokenize
from nltk.metrics import *

now = datetime.datetime.now()
WORD = re.compile(r'\w+')

stemmer = stem.PorterStemmer()

alpha = 0.9
beta = 3.2
gamma = 0.5


can = 20
top = 4
andup = 0   #Always set to 0 unless testing higher values
cutoff = 35
edgecorrection = 10
samethresh = .0000001

ncut = 2
thereval = 0.5
fromfile = True

nweight = 0.11
cosweight = 9
genweight = 1.7
mweight = 1
fuzzweight = 0.01

##This function determines cosine simularity
##It's based off the code found on stack overflow at this link:
#https://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def the_thing(candidates, currentlist, possiblelists,  taken, pos):
    #print("pos " + str(pos))
    #print(currentlist)
    #base case:
    if pos == len(candidates) - 1:
        # print("at base")
        # print(candidates[pos][1])
        for item in candidates[pos][1]:
            if not taken[item]:
                # print("Hey we got past if")
                # print("still okay")
                # print(currentlist[pos])
                # print(candidates[pos][0])
                # print("Here nowsss")
                currentlist[pos] = (item, candidates[pos][0])
                # print("2")
                newlist = currentlist
                # print("3")
                possiblelists.append(newlist)
                # print("append new list")
    else:
        # print("not at base")
        for item in candidates[pos][1]:
            # print(candidates[pos][1])
            if not taken[item]:
                currentlist[pos] = (item, candidates[pos][0])
                taken[item] = True
                the_thing(candidates, currentlist, possiblelists, taken, pos+1)
                taken[item] = False
    # print("return")
    return

def normalize(s):
    words = tokenize.wordpunct_tokenize(s.lower().strip())
    return ' '.join([stemmer.stem(w) for w in words])
 
def fuzzy_distance(s1, s2):
    print(edit_distance("rain", "shine"))
    edit_distance("sun", "shine")


def cmp_2(candidates):
    return len(candidates[1])

def perms(wp, rp, container):
  #  print("here")
    candidates = []
    filler = []

    fill = -1
    if len(rp) > len(wp):
        for i in range(len(rp), len(wp)):
            filler.append(fill)
            fill-=1

    tryn = 0

    for rpos in rp:
        wordcandidates = []
        for wpos in wp:
            if rpos == None or abs(rpos - wpos) < can:
                wordcandidates.append(wpos)
            for fill in filler:
                wordcandidates.append(fill)


        candidates.append((rpos ,wordcandidates))
        candidates = sorted(candidates, key=cmp_2, reverse=True)

   


    taken = dict()
    for wpos in wp:
        taken[wpos] = False
        #print("taken: " + str(wpos))

    possiblelists = []
    currentlist = []
    pos = 0
    #print("len rp? " + str(len(rp)))
    for i in range(0, len(rp)):
      #  print("hey!")
        currentlist.append(None)

    ######USE RECURSION########
   # print("enter here")

   # print("lenrp " + str(len(rp)))
   # print("lencand " + str(len(candidates)))
    #print("lencurr " + str(len(currentlist)))
    # print("not bad so far")
    the_thing(candidates, currentlist, possiblelists, taken, 0)

    for plist in possiblelists:
        for i in range(0, len(plist)):
            if plist[i][0] < 0:
                del plist[i]
    # print("possiblielists")
    # print(possiblelists)

    for plis in possiblelists:
        if None not in plis:
            container.append(plis)
   # print("finihere"

    #print(possiblelists)
   # print("pls")








def scoring(al, h1, ref):
    m = 0
    t = 0
    r = 0
    #print(al)
    for i in range(0, len(al)):
        pos = al[i]
        if pos == None:
            al[i] = (i, None)
        elif pos[1] != None:
            m+=1

    if m == 0:
        return 0

    for word in h1:
        t+=1

    for word in ref:
        r+=1

    P = m/float(t)
    R = m/float(r)



    Fmean = (P*R)/((alpha*P)+((1-alpha)*R))

    ch = 1
    lastr = al[0][1]
    index = 0
    while lastr == None:
        index+=1
        lastr = al[index][1]


    for i in range(index, len(al)):
        pos = al[i]
        if pos[1] == None:
            lastr = -5
        else:
            if pos[1] != lastr + 1:
                ch+=1
            lastr = pos[1]

    frag = ch/m
    Pen = gamma*((frag)**beta)

    score = (1-Pen)*Fmean

    return score




def cross_count(al):
    count = 0
    for i in range(0, len(al) - 1):
        for j in range(i+1, len(al)):
            ab = al[i]
            xy = al[j]
            if ab!= None and xy!= None and None not in xy and None not in ab:
                product = (ab[0]-xy[0])*(ab[1]-xy[1])
                if product < 0:
                    count+=1

    return count

def cmp_key(al):
    #print(al)
    return cross_count(al)


def gen_alignment (h, r):

    h1 = []
    ref = []

    for word in h:
        h1.append(word.lower())

    for word in r:
        ref.append(word.lower())

    h1_repeats = []
    ref_repeats = []
    h1_cheat = []
    ref_cheat = []
    h1_word_count = dict()
    ref_word_count = dict()



    for i in range (0, len(h1)):
        try:
            h1_word_count[h1[i]].append(i)
        except:
            h1_word_count[h1[i]]=[]
            h1_word_count[h1[i]].append(i)

    for i in range(0, len(ref)):
        try:
            ref_word_count[ref[i]].append(i)
        except:
            ref_word_count[ref[i]]=[]
            ref_word_count[ref[i]].append(i)

    for i in range (0, len(h1)):
        if len(h1_word_count[h1[i]]) > 1 and h1[i] not in h1_cheat:
            h1_cheat.append(h1[i])
            h1_repeats.append((h1[i], h1_word_count[h1[i]] ))

    index = -1
    for i in range(0, len(ref)):
        if len(ref_word_count[ref[i]]) > 1 and ref[i] not in ref_cheat:
            ref_cheat.append(ref[i])
            ref_repeats.append((ref[i], ref_word_count[ref[i]]))
            index = i

    base = []

    for i in range(0, len(h1)):
        if len(h1_word_count[h1[i]]) < 2:
            try:
                base.append((i, ref_word_count[h1[i]][0]))
            except:
                base.append((i , None ))
        else:
                base.append(None)

    # print("sentence 1")
    # print(h1)
    # print("Sentence 2")
    # print(ref)
    # print("Base")
    # print(base)

    baselist = []
    baselist.append(base)
    a_list = []
    a_list = baselist
    track = 0
    test = 1
    if len(h1_repeats) == 0:
        test = 0

    #for i in range(0, test):
    for i in range(0, len(h1_repeats)):
        # print("word")
        current_word = h1_repeats[i]
        # print("currentwordthing")
        # print(current_word)
        appendedlist = []
        for unitlist in a_list:
           # print("gooby")
            word_positions = h1_word_count[current_word[0]]
            
            try:
                # print("haaa")
                ref_positions = ref_word_count[current_word[0]]
                

                combos1 = list(itertools.permutations(word_positions))
                
                
                combosref = list(itertools.permutations(ref_positions))
                #print("tough2")
                # print("comobs1")
                # print(combos1)
                # print("combos2")
                # print(combosref)

                wp = word_positions[:]
                rp = ref_positions[:]

                

                while len(wp) < len(rp):
                    wp.append(None)
                    

                while len(rp) < len(wp):
                    rp.append(None)

                #print(word_positions)

                    
                
                #print("big")
                # print("wp")
                # print(wp)
                # print("rp")
                # print(rp)
                # print("perms")
                # perms(wp, rp)
                # print("wp")
                # print(wp)
                # print("rp")
                # print(rp)
                #container = [zip(x,rp) for x in itertools.permutations(wp,len(rp))]
                try:
                    # print("wp")
                    # print(wp)
                    # print("rp")
                    # print(rp)
                    container = []
                    # print("wp rp")
                    # print(wp)
                    # print(rp)
                    # perms(wp, rp, container)
                    # print(container)
                    # print("container")
                    # print(container)
                    # print("ah")
                except:
                     print("ERROR1")
                #print("bigemd")
                # print("word_positions")
                # print(wp)
                # print("wp")
                # print("ref positions")
                # print(rp)
                #print(container)

                for item in container:
                    #print("item")
                    newlist = unitlist
                    for pos in item:
                        # print("pos")
                        # print(pos)
                        if pos[0] == None:
                            pass
                        else:
                            newlist[pos[0]] = pos
                    # print("Actual New list")
                    # print(newlist)
                    appendedlist.append(newlist)
                # print("old list")
                # print(unitlist)
                # print("newlist")
                # print(newlist)

            except:
                # print("gaasas")
                #attach all to none
                newlist = unitlist
                # print(word_positions)
                for word_pos in word_positions:
                    newlist[word_pos] = (word_pos, None)
                appendedlist.append(newlist)

                


        if len(appendedlist) > 0:
            a_list = appendedlist
            if len(a_list) > top:
                a_list = sorted(a_list, key=cmp_key, reverse=True)
                finallist = []
                for i in range(0, top):
                    finallist.append(a_list[i])
                a_list = finallist
            #print("AppendedList")
            #print(appendedlist)
        track += 1

    # deletelist = []

    # for i in range(0, len(a_list)):
    #     for j in range(0, len(a_list[i])):
    #         if a_list[i][j] == None:
    #             deletelist.append(a_list[i])
    #             break

    # adjust = 0
    # for i in range(0, len(deletelist)):
    #     del a_list[i-adjust]
    #     adjust+=1


    current_score = cross_count(a_list[0])
    current_index = 0
    for i in range(0, len(a_list)):
        score = cross_count(a_list[i])
        if score < current_score:
            current_score = score
            current_index = i

    return a_list[current_index]







    # if index > 0:
    #     # print("test")
    #     #result = itertools.combinations(ref_word_count[ref[i]], len(ref_word_count[ref[i]]))
    #     thing = ref_word_count[ref[index]]
    #     # els = [list(x) for x in itertools.combinations(thing, len(thing))]
    #     els = list(itertools.permutations(thing))
    #     # print(thing)
    #     # print(els)

    # # print("gen allign")
    # # print(ref)
    # # print(ref_repeats)





 
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
    f = open('nscores.txt', 'r')
    if not fromfile:
        f2 = open(str(now) + 'nscores.txt', 'w')
    lines = list(f)
    nscorelist1 = []
    nscorelist2 = []
    for i in range(0,len(lines)):
        if i%2 == 0:
            nscorelist1.append(float(lines[i]))
        else:
            nscorelist2.append(float(lines[i]))

 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    counter = -1
    limit = 10
    
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        counter+=1
        # print"sentence: ",
        # print counter
        sys.stderr.write(str(counter) + "\n")
        if counter >= andup:
            rset = set(ref)
            # print("REF")
            # print(ref)
            # print("h1")
            # print(h1)
            # print(h1)
            # print(h2)
            # print(ref)
            pieces = (len(ref)/cutoff) + 1
            h1list = []
            h2list = []
            reflist = []

            amt = len(h1)/pieces
            cnt = 0
            for i in range(0, pieces):
                if cnt - edgecorrection < 0:
                    if cnt+amt < len(h1):
                        h1list.append(h1[cnt:cnt+amt])
                    else:
                        h1list.append(h1[cnt:len(h1)-1])
                    cnt+=amt
                else:
                    if cnt+amt < len(h1):
                        h1list.append(h1[cnt-edgecorrection:cnt+amt])
                    else:
                        h1list.append(h1[cnt-edgecorrection:len(h1)-1])
                cnt+=amt

            amt = len(h2)/pieces
            cnt = 0
            for i in range(0, pieces):
                if cnt - edgecorrection < 0:
                    if cnt+amt < len(h2):
                        h2list.append(h2[cnt:cnt+amt])
                    else:
                        h2list.append(h2[cnt:len(h2)-1])
                    cnt+=amt
                else:
                    if cnt+amt < len(h2):
                        h2list.append(h2[cnt-edgecorrection:cnt+amt])
                    else:
                        h2list.append(h2[cnt-edgecorrection:len(h2)-1])
                cnt+=amt

            amt = len(ref)/pieces
            cnt = 0
            for i in range(0, pieces):
                if cnt - edgecorrection < 0:
                    if cnt+amt < len(ref):
                        reflist.append(ref[cnt:cnt+amt])
                    else:
                        reflist.append(ref[cnt:len(ref)-1])
                    cnt+=amt
                else:
                    if cnt+amt < len(ref):
                        reflist.append(ref[cnt-edgecorrection:cnt+amt])
                    else:
                        reflist.append(ref[cnt-edgecorrection:len(ref)-1])
                cnt+=amt

            h1score = 0
            h2score = 0

            for i in range(0, pieces):
                a1 = gen_alignment(h1list[i], reflist[i])
                a2 = gen_alignment(h2list[i], reflist[i])
                s1 = scoring(a1, h1list[i], reflist[i])
                s2 = scoring(a2, h2list[i], reflist[i])
                #############WORD NET STUFF################
                if not fromfile:
                    h1misses = []
                    h2misses = []
                    for pos in a1:
                        if pos[1] == None and pos[0] != None:
                            h1misses.append((h1[pos[0]], pos[0]))

                    for pos in a2:
                        if pos[1] == None and pos[0] != None:
                            h2misses.append((h2[pos[0]], pos[0]))

                    nscore1 = 0
                    nscore2 = 0
                    refsy = []
                    for w in ref:
                        try:
                            rword = Word(w)
                            rim = rword.synsets[:ncut]
                            for r in rim:
                                refsy.append(r)
                        except UnicodeDecodeError:
                            pass

                    for thing in h1misses:
                        try:
                            
                            word = Word(thing[0])
                            words = word.synsets[:ncut]
                            syslist = []
                            for w in words:
                                syslist.append(w)

                            highestx = 0
                            highesty = 0
                            currentsim = 0
                            for i in range(0, len(syslist)):
                                for j in range(0, len(refsy)):
                                    if syslist[i].path_similarity(refsy[j]) > currentsim:
                                        currentsim = syslist[i].path_similarity(refsy[j])
                            if currentsim == 1:
                                nscore1+= thereval
                            else:
                                nscore1+=currentsim
                        except UnicodeDecodeError:
                            pass

                        
                    for thing in h2misses:
                        try:
                            word = Word(thing[0])
                            words = word.synsets[:ncut]
                            syslist = []
                            for w in words:
                                syslist.append(w)

                            highestx = 0
                            highesty = 0
                            currentsim = 0
                            for i in range(0, len(syslist)):
                                for j in range(0, len(refsy)):
                                    if syslist[i].path_similarity(refsy[j]) > currentsim:
                                        currentsim = syslist[i].path_similarity(refsy[j])

                            if currentsim == 1:
                                nscore2+= thereval
                            else:
                                nscore2+=currentsim
                        except UnicodeDecodeError:
                            pass

                    # # print("nscore1: " + str(nscore1))
                    # # print("nscore2: " + str(nscore2))

                    f2.write(str(nscore1)+"\n")
                    f2.write(str(nscore2)+"\n")
                else:


                    nscore1 = nscorelist1[counter]
                    nscore2 = nscorelist2[counter]

                



                #############Cosine Simularity###############
                refstr = " ".join(ref)
                h1str = " ".join(h1)
                h2str = " ".join(h2)

                vref = text_to_vector(refstr)
                v1 = text_to_vector(h1str)
                v2 = text_to_vector(h2str)

                cos1 = get_cosine(v1, vref)
                cos2 = get_cosine(v2, vref)


                ##############Gensim simularity###############
                ##Based on this tutorial:
                ##https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
                genscore1 = 0
                genscore2 = 0
                try:

                    raw_documents = [h1str, h2str]
                    # print("Number of documents:",len(raw_documents))
                    gen_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in raw_documents]
                    # print(gen_docs)
                    dictionary = gensim.corpora.Dictionary(gen_docs)
                    # print(dictionary[5])
                    # print(dictionary.token2id['cities'])
                    # print("Number of words in dictionary:",len(dictionary))
                    # for i in range(len(dictionary)):
                    #     print(i, dictionary[i])
                    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
                    # print(corpus)
                    tf_idf = gensim.models.TfidfModel(corpus)
                    # print(tf_idf)
                    s = 0
                    for i in corpus:
                        s += len(i)
                    # print(s)
                    sims = gensim.similarities.Similarity('.',tf_idf[corpus],num_features=len(dictionary))
                    # print(sims)
                    # print(type(sims))
                    query_doc = [w.lower() for w in word_tokenize(refstr)]
                    query_doc_bow = dictionary.doc2bow(query_doc)
                    query_doc_tf_idf = tf_idf[query_doc_bow]
                    
                    genscore1 += sims[query_doc_tf_idf][0]
                    genscore2 += sims[query_doc_tf_idf][1]

                except UnicodeDecodeError:
                    pass




                #################Fuzzy Match################
                fuzzyscore1 = edit_distance(refstr, h1str)
                fuzzyscore2 = edit_distance(refstr, h2str)

                # print(sims[query_doc_tf_idf])
                h1score += (mweight*((1.0/pieces)*s1)) + (nweight*nscore1) + (cosweight*cos1) + (genweight*genscore1) + (fuzzweight*fuzzyscore1)
                h2score += (mweight*((1.0/pieces)*s2)) + (nweight*nscore2) + (cosweight*cos2) + (genweight*genscore2) + (fuzzweight*fuzzyscore2)

            # al1 = gen_alignment(h1, ref)
            # print("al1 done")
            # al2 = gen_alignment(h2, ref)
            # print("h1")
            # print(h1)
            # print("h2")
            # print(h2)
            # print("ref")
            # print(ref)
            # print("al1")
            # print(al1)
            # print("al2")
            # print(al2)
            # print("al1 al2")
            # print(al1)
            # print(al2)
            # score1 = scoring(al1, h1, ref)
            # score2 = scoring(al2, h2, ref)
            # print("scores")
            # print(score1)
            # print(score2)
            # print("fini")
            h1_match = word_matches(h1, rset)
            h2_match = word_matches(h2, rset)
            # if counter > limit:
            #     break
            # print("Normal h1: " + str(score1))
            # print("normal h2: " + str(score2))
            # print("S1: " + str(h1score))
            # print("S2: " + str(h2score))
            # print (nscore1)
            # print (nscore2)
            if abs(h1score - h2score) < samethresh:
                print(0)
            elif h1score > h2score:
                print(1)
            else:
                print(-1)
            
            




# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
