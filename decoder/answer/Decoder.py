#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

opts.k = 4
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]




# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]
# values for the model parameter
dd = 5
nn = -4
beta = 2

gooby = tm[("de", "ce")]
#print(len(gooby))
#print(gooby)

class state:
  def __init__(self, e1, e2, b, r, alpha):
    self.e1 = e1
    self.e2 = e2
    self.b = b
    self.r = r
    self.alpha = alpha

def beam(Q, beta):
  #print("IN BEAM")
  alphastar = -99999
  for q in Q:
    if q.alpha > alphastar:
      alphastar = q.alpha

  qbeam = []
  #print(alphastar)
  for q in Q:
    #printstate(q)
    if q.alpha > alphastar - beta:
      qbeam.append(q)
  #print("LEAVING BEAM")
  return qbeam



def ph(q, sentence, distortion, tm):
  #print("IN PH")
  #printstate(q)
  phrases = []
  
  for s in range(0, len(sentence)):
    for t in range(s, len(sentence)):
      if match(s, t, q.b):
        #print("s:" + str(s) + "t:" + str(t) + "break")
        break
      tup3 = []
      for i in range(s, t+1):
        tup3.append(sentence[i])
      ##print(tup3)
      tup = tuple(tup3)
      try:
    #//#print("Appending")
        for i in range(0,len(tm[tup])):
          phrases.append((s, t, tm[tup][i]))
          #print(tup)
          #print(tm[tup][i])
      except:
        pass
  #printstate(q)
  #print("LEAVING PH")
  return phrases



def match(s, t, b):
  for i in range(s, t):
    if b[i] == True:
      return True
  return False

def next(q, p, lm, nn ):
  #print("IN NEXT")
  #print("q:")
  #printstate(q)
  pdwords = p[2][0]
  pwords = pdwords.split()
  newe1 = ""
  newe2 = ""
  if len(pwords) < 1:
    newe1 = q.e1
    newe2 = q.e2
  elif len(pwords) < 2:
    newe1 = q.e2
    newe2 = pwords[len(pwords) - 1]
  else:
    newe1 = pwords[len(pwords) - 1]
    newe2 = pwords[len(pwords) - 2]

  newb = []
  for i in range(0, len(q.b)):
    if q.b[i]:
      newb.append(True)
    else:
      newb.append(False)

  for i in range(p[0], p[1] + 1):
    newb[i] = True

  newr = p[1]
  ##Fun part get h
  logprob = 0.0
  lm_state = lm.begin()
  for word in pwords:
    (lm_state, word_logprob) = lm.score(lm_state, word)
    logprob += word_logprob
    logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
  #print("q.alpha: " + str(q.alpha) + " g(): " + str(p[2][1]) + " logprob: " + str(logprob) + "distortion: " + str(nn*abs(q.r + 1 - p[0])) )
  #All together
  newalpha = q.alpha + p[2][1] + logprob + nn*abs(q.r + 1 - p[0])

  newq = state(newe1, newe2, newb, newr, newalpha)
  #print("newq")
  #printstate(newq)
  #print("LEACVING NEXT")
  return newq

def printstate(q):
  print("#printing state")
  print("e1: " + str(q.e1) + " " + "e2: " + str(q.e2))
  print("r: " + str(q.r))
  print("alpha: " + str(q.alpha))
  print(q.b)

def eq(q1, q2):
  if q1.e1 != q2.e1:
    return False
  if q2.e1 != q2.e2:
    return False
  if q1.r != q2.r:
    return False

  for i in range(0,len(q1.b)):
    if q1.b[i] != q2.b[i]:
      return False
  return True

def Add(Q, qprime, q, p, bp):
  #print("In ADD")
  if Q == None:
    Q = []
  for i in range(0, len(Q)):
    if eq(Q[i], qprime):
      if qprime.alpha > Q[i].alpha:
        Q[i] = qprime
        #printstate(qprime)
        #set bpointer
        bp[qprime] = (q, p)
        #print("LEAVing1 ADD")
        return Q
      else:
        #printstate(qprime)
        #print("LEAVing2 ADD")
        return Q

  Q.append(qprime)
  bp[qprime] = (q, p)
      #set bpointer
  #printstate(qprime)
  #print("LEAVing3 ADD")
  return Q


def getlength(b):
  counter = 0
  for i in range(0, len(b)):
    if b[i]:
      counter += 1

  return counter


#print("heeeey")



for sentence in range(0, len(french)):
  #print("//////////////////////////")
  bp = dict()

  n = len(french[sentence])
  b = []
  Q = []
  for i in range(0,n):
    b.append(False)
  qin = state("*", "*", b, -1, -1)
  q0 = []
  q0.append(qin)
  Q.append(q0)

  for i in range(1,n+1):
    Q.append(None)

  index = 0
  for i in range(0, n): ##change 1 to n later
    
    
    index = i
    #print("i: " + str(i))
    #print(Q[i])
    #print(len(french[sentence]))
    #print("Q[" + str(i) + "]")
    #print(type(Q[i])) 
    for q in beam(Q[i], beta):
      #print("III: " + str(i))
      for p in ph(q, french[sentence], dd, tm):
        newq = next(q, p, lm, nn)
        j = getlength(newq.b)
        #print("I: " + str(i) + " J: " + str(j))
        #print("Old q")
        #printstate(q)
        #print("new q")
        #printstate(newq)
        Q[j] = Add(Q[j], newq, q, p, bp)
        #print(type(Q[j]))
        index = j


  best = 0
  index = len(french[sentence])
  for i in range(0, len(Q[index])):
    if Q[index][i].alpha > Q[index][best].alpha:
      best = i

  #print("Best " + str(best))

  a = Q[index][best]
  (a,b) = bp[a]
  wholesentence = []
  while True:
    wholesentence.insert(0, b[2][0])
    try:
      (a,b) = bp[a]
    except:
      break

  first = True
  for word in wholesentence:
    if first:
      word = word.capitalize()
      first = False
    print word,
  print("")


# #print("b")
# #print(b)
  







