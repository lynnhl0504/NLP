import sys, codecs, optparse, os, heapq, math

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0 
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class Entry:
    def __init__(self, word, start, log, bpointer, mark):
        self.word = word
        self.start = start
        self.log = log
        self.bpointer = bpointer
        self.mark = mark
  

# the default segmenter does not use any probabilities, but you could ...
#Pw  = Pdist(opts.counts1w)
Pw  = Pdist(opts.counts1w)
Pw2 = Pdist(opts.counts2w)
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts

#initialize heap

def heapinsert(entries, minput, h, position, entry, pw, pw2):
	###print("In heap insert")
	###print("MIN POSITION: ")
	###print(minput[position])
	###print("position:" + str(position))

	for line in entries:
		
		pos = 0
		good = True
		uniword = line[0]
		uniword.strip()
		while pos < len(uniword):
			if pos+position >= len(minput) or uniword[pos] != minput[pos+position]:
				good = False
				break
			pos+=1
		#If good at to the heap
		if good and len(uniword) > 0:
			#probuni = math.log10(number/85075.0);

			probuni = pw(line[0])

			if pw2(line[0]) == None:
				 probuni = pw(line[0])

			#if probuni == None:
			#	probuni = pw2(line[0])

			#if probuni != None:
			#	probuni = pw2(line[0]) * pw(line[0])

			entry = Entry(line[0], position, probuni, position-1, False)	#### number is put in as log for now
			#print("Entry1: " + entry.word + "entry1done")
			match = False;
			for item in h:	##Check if entry is already in the heap
				if item[1].word == entry.word:
					match = True
					break
			if not match:
				heapq.heappush(h, ( position, entry ))

				##print("Push 1: " + entry.word)
		# if good and len(uniword) == 0:
		# 	probuni = 0
		# 	entry = Entry(line[0], position, probuni, entry)	#### number is put in as log for now
		# 	###print("Entry2: " + entry.word + "entry2done")
		# 	match = False;
		# 	for item in h:	##Check if entry is already in the heap
		# 		if item[1].word == entry.word:
		# 			match = True
		# 			break
		# 	if not match:
		# 		heapq.heappush(h, ( position, entry ))
		# 		###print("Push 2: " + entry.word)


####print("hola")
#heap
h = []
entries = []
position = 0
with open(opts.input) as largeinput:
	with open(opts.counts1w) as wordlist1:
		with open(opts.counts2w) as wordlist2:
			####print("Here!")
			#Match each word in word list to pos0 in input
			minput = ""
			#Make entire file utf-8
			# for line in largeinput:
			# 	utf8line = unicode(line.strip(), 'utf-8')
			# 	minput+=utf8line
			minput = unicode(largeinput.read().strip(), 'utf-8')
			#Match each word
			#Seperate mandarin word from number
			for line in wordlist1:
				word = ""
				number = 0
				for i in line:
					if i.isdigit():
						number = int(i)
						##print("Number:" + str(number))
					else:
						word+=i
					####print("Out of if else")
				uniword = unicode(word.strip(), 'utf-8')
				# ###print(uniword)
				entries.append((uniword, number))
###########################Comment out for just unigram################################
			for line in wordlist2:
				word = ""
				number = 0
				spacefound = False
				for i in line:
					if i.isdigit():
						number = int(i)
						##print("Number:" + str(number))
					elif i == " ":
						if spacefound:
							break
						else:
							spacefound = True
							word+=i
					elif i == "\n":
						break
					else:
						word+=i

					####print("Out of if else")
				uniword = unicode(word, 'utf-8')
				####print(uniword)
				entries.append((uniword, number))
#########################################################################################
			###print("Calling heap insert 1")
			heapinsert(entries, minput, h, position, None, Pw, Pw2)
			####print("here4")
				#######################################

				###Done making heap###
			####print ("Pw2:" + Pw2)
			###FILL IN CHART#####

			# while len(h) > 0:
			# 	entry = heapq.heappop(h)[1]
			# 	###print(entry.word)
			chart = [None] * len(minput)
			while len(h) > 0:
				entry = heapq.heappop(h)[1]
				#print("We're on "  + entry.word)
				#print("Position: " + str(position))
				endindex = entry.start + len(entry.word) - 1
				if chart[endindex] != None:
					preventry = chart[endindex]
					##entry has higher probability
					amount = len(preventry.word.strip())
					cycle = entry
					prob = 1
					while amount > 0:
						prob = prob * cycle.log
						cycle = chart[cycle.bpointer]
						amount = amount - 1

					if prob > preventry.log:
						chart[endindex] = entry
						#print("higher prob. preventry: " + str(preventry.log) + "< Entry: " + str(entry.log))
						#print(preventry.word + " " + entry.word)
						#print(str(Pw(preventry.word)) + " " + str(Pw(entry.word)))
						##print("1chart[" + str(endindex) + "]= " + entry.word + " prob: " + str(entry.log))
				else:
					chart[endindex] = entry
					##print("2chart[" + str(endindex) + "]= " + entry.word + " prob: " + str(entry.log))
					###print("added to chart [" + entry.word + "]" )
				position = endindex + 1
				###print("calling heap insert 2")
				heapinsert(entries, minput, h, position, entry, Pw, Pw2)
				if len(h) == 0 and endindex < len(minput)- 1:
					entry = Entry(minput[position], position, 0, position-1, True)
					#print("Name not found" + entry.word)
					heapq.heappush(h, ( position, entry ))


gooby = chart[len(minput)-1]
list = []
while gooby != None:
	list.append(gooby)
	if gooby.bpointer != -1:
		gooby = chart[gooby.bpointer]
	else:
		gooby = None

i = len(list) - 1

while i >= 0:
	# if list[i-1].word[0] != list[i].word[0]:
		####print(list[i].word)
		####print (list[i].word + " ")
	if (list[i-1].word == "\n" or list[i].word == "\n" or (list[i-1].mark and list[i].mark)):
		sys.stdout.write (list[i].word)
	else:
		sys.stdout.write (list[i].word + " ")
	i = i - 1

#print ("///////////////////////////////////////")
# i = 0
# while (i < len(chart)):
# 	if chart[i] != None:
# 		#print("chart[" + str(i) + "] = " + chart[i].word)


# 	i = i + 1






# with open(opts.input) as f:
#     for line in f:
#         utf8line = unicode(line.strip(), 'utf-8')
#         output = [i for i in utf8line]  # segmentation is one word per character in the input
#         ####print " ".join(output)
#         ###print " ".join(output)

sys.stdout = old






















