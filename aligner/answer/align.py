#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


#Initialization
k = 0
fvocab = defaultdict(int)
fvcount = 0
for sentencePair in bitext:
  for frenchWord in sentencePair[0]:
    if fvocab[frenchWord] == 0:
      fvocab[frenchWord] = 1
      fvcount += 1
    else:
      fvocab[frenchWord] += 1

 ###################ADDDED THIS####################
jk = 0
fkvocab = defaultdict(int)
fkvcount = 0
for sentencePair in bitext:
  for englishWord in sentencePair[1]:
    if fkvocab[englishWord] == 0:
      fkvocab[englishWord] = 1
      fkvcount += 1
    else:
      fkvocab[englishWord] += 1

#Now we have how many different french words there
#Now we must go through all french/english word pairs 
#We're only going to do pairs that appear in sentences

#Intitialize t0

t = []
t2 = []
t.append(defaultdict(float))#reference as t[0]["word"]
t2.append(defaultdict(float))
for sentencePair in bitext:
  for frenchWord in sentencePair[0]:
    for englishWord in sentencePair[1]:
      t[0][(frenchWord, englishWord)] = 1.0 / fvcount
      #JUST ADDED THIS v
      t2[0][(englishWord, frenchWord)] = 1.0 / fkvcount

# for sentencePair in bitext:
#   for englishWord in sentencePair[1]:
#     for frenchWord in sentencePair[0]:
#       t2[0][(englishWord, frenchWord)] = 1.0 / fkvcount
#       print (t2[0][(englishWord, frenchWord)])


#Main Loop
while True:
  t.append(defaultdict(float))
  #added this v
  t2.append(defaultdict(float))
  k += 1
  #Define E count and FE count and set them to 0
  E_count = defaultdict(int)
  F_count = defaultdict(int)
  FE_count = defaultdict(int)
  #JUST ADDED THIS v
  EF_count = defaultdict(int)
  
  for sentencePair in bitext:
    for englishWord in sentencePair[1]:
      E_count[englishWord] = 0
      for frenchWord in sentencePair[0]:
        FE_count[(frenchWord, englishWord)] = 0
        #JUST ADDED THIS v
        EF_count[(englishWord, frenchWord)] = 0
        F_count[frenchWord] = 0

  #Go through each sentence pait in D
  for sentencePair in bitext:
    for frenchWord in sentencePair[0]:
      Z = 0
      X = 0
      for englishWord in sentencePair[1]:
        Z += t[k-1][(frenchWord, englishWord)]
        X += t[k-1][(englishWord, frenchWord)]
      #
      for englishWord in sentencePair[1]:
        c = t[k-1][(frenchWord, englishWord)]/Z
        #d = t[k-1][(englishWord, frenchWord)]/Z

       #########################ADDED THIS##########################
    for englishWord in sentencePair[1]:
      Z2 = 0
      
      for frenchWord in sentencePair[0]:
        Z2 += t2[k-1][(englishWord, frenchWord)]
        #X2 += t2[k-1][(frenchWord, englishWord)]
      #
      for frenchWord in sentencePair[0]:
        c2 = t2[k-1][(englishWord, frenchWord)]/Z2
        #d2 = t2[k-1][(frenchWord, englishWord)]/Z2 
         ###########################################
        
        FE_count[(frenchWord, englishWord)] += c 
          #FE_count[(frenchWord, englishWord)] += (c + d) /2
        EF_count[(englishWord, frenchWord)] += c2 

        #E_count[englishWord] += c
        E_count[englishWord] += c 
        F_count[frenchWord] += c2 

  #Go through each f,e in count
  for sentencePair in bitext:
    for frenchWord in sentencePair[0]:
      for englishWord in sentencePair[1]:
        t[k][(frenchWord, englishWord)] = FE_count[(frenchWord, englishWord)]/E_count[englishWord]
        ##aded this v
        if F_count[englishWord] != 0:
          t2[k][(englishWord, frenchWord)] = EF_count[(englishWord, frenchWord)]/F_count[englishWord]
        if F_count[englishWord] == 0:  
          t2[k][(englishWord, frenchWord)] = t[k][(frenchWord, englishWord)]
        
  #Run convergence tests
  #if Good then break
  if k == 5:
    break

for englishWord in sentencePair[1]:
  for frenchWord in sentencePair[0]:
    t[k][(frenchWord, englishWord)] = (t[k][(frenchWord, englishWord)] + t2[k][(frenchWord, englishWord)] )/2



for sentencePair in bitext:
  french = 0
  for frenchWord in sentencePair[0]:
    bestp = 0
    bestj = 0
    english = 0
    for englishWord in sentencePair[1]:
      if t[k][(frenchWord, englishWord)] > bestp:
        bestp = t[k][(frenchWord, englishWord)]
        bestj = english
      english+=1
    #Allign frenchword to english word using their positions in the sentence
    if french < len(sentencePair[0]) - 1:
      sys.stdout.write(str(french) + "-" + str(bestj) + " ")
    else:
      sys.stdout.write(str(french) + "-" + str(bestj) + "\n")
    french+=1



# BELOW IS DEFAULT
# f_count = defaultdict(int)
# e_count = defaultdict(int)
# fe_count = defaultdict(int)
# for (n, (f, e)) in enumerate(bitext):
#   for f_i in set(f):
#     f_count[f_i] += 1
#     for e_j in set(e):
#       fe_count[(f_i,e_j)] += 1
#   for e_j in set(e):
#     e_count[e_j] += 1
#   if n % 500 == 0:
#     sys.stderr.write(".")

# print("hi appears " + str(e_count[("hi","nous")]) + "times.")  


# dice = defaultdict(int)
# for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
#   dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
#   if k % 5000 == 0:
#     sys.stderr.write(".")
# sys.stderr.write("\n")


# for (f, e) in bitext:
#   for (i, f_i) in enumerate(f): 
#     for (j, e_j) in enumerate(e):
#       if dice[(f_i,e_j)] >= opts.threshold:
#         sys.stdout.write("%i-%i " % (i,j))
#   sys.stdout.write("\n")
