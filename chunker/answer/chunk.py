"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys , optparse, os

from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)     
    i = 0
    j = 0
    while i < len(train_data):
        while j < len(train_data[i][1]):
            feat = train_data[i][1][j]
            for tag in tagset:
                feat_vec[feat, tag] = 0
            j+=1
        i+=1


    gooby = perc.perc_test(feat_vec, train_data[0][0], train_data[0][1], tagset, tagset[0])
    #print(train_data[0][0])
    #print(getDefault(train_data[0][0]))
    for i in range(1,numepochs):
    # for i in range(1,numepochs):
        for j in range(0, len(train_data)-1):
            for k in range(0,len(train_data[j])-1):
                #Get Default Tag
                defaultTag = getDefault(train_data[j][0])
                z = perc.perc_test(feat_vec, train_data[j][0], train_data[j][1], tagset, defaultTag)
                # print(z)

                #compare z to t
                t = []
                for line in train_data[j][0]:
                    t.append(line.split()[2])
                #update weights if z and t are not the same
                for a in range (0,len(z)-1): 
                    if z[a] != t[a]:
                        for b in range (20*a, 20*a + 19):
                            try:
                                feat_vec[(train_data[j][1][b], t[a])] += 1
                                feat_vec[(train_data[j][1][b], z[a])] -= 1
                            except:
                                print("Error")
                                print("j = " + str(j) + " b = " + str(b) + " a = " + str(a))
                                print("len(train_data) = " + str(len(train_data)))
                                print("len(train_data[j][1]) = " + str(len(train_data[j][1])))
                                print("len(t) = " + str(len(t)))
                                print("len(z) = " + str(len(z)))
                                print("train_data[j][1][738] = " + train_data[j][1][738])
                                print("train_data[j][1][739] = " + train_data[j][1][739])
                                exit()


                #if z != t:
                    #feat_vec += feat_vec + perc.perc_test(feat_vec, train_data[j][0], train_data[j][1], tagset, defaultTag)

    return feat_vec                





def getDefault(sentence):
    wordcounts = {}
    for line in sentence:
        word = line.split()[2]
        wordcounts[word] = 0

    maximum = (0, None)

    for line in sentence:
        word = line.split()[2]
        wordcounts[word] += 1
        if (maximum[0] < wordcounts[word]):
            maximum = (wordcounts[word], word)

    return maximum[1]





if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

