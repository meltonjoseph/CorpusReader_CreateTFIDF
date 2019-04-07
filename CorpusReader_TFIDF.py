"""
Joseph Melton
NLP
2/12/2019
"""
import nltk
from math import log, sqrt
from nltk import *
from nltk.corpus import stopwords

class CorpusReaderTFIDF:

    '''
    Constructor that takes in a corpus, tf, idf, stopword path, stemmer to use, and whether
    or not to ignore case
    '''
    def __init__(self, corpus=None, tf="raw", idf="base", stopword=set(stopwords.words('english')),
                 stemmer=nltk.PorterStemmer(), ignorecase="yes"):

        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.tfidf = []
        self.matrixTF = []
        self.vectorIDF = []
        self.fileids = corpus.fileids()
        self.allWords = []
        self.dictAllWords = dict()
        self.postProccesCorpus = dict()
        self.ignorecase = ignorecase
        self.stopword = stopword
        self.stemmer = stemmer

        # Checking if we use default english stopword or one passed in by user
        if stopword == "none":
            self.stopword = list()
        elif type(stopword) == str:
            text_file = open(stopword)
            self.stopword = set(text_file.read().split(' '))
        else:
            self.stopword = stopword

        # Checking if we need to ignore case or not
        if ignorecase == "yes":
            self.allWords = [a.lower() for a in self.allWords]
        # Doing the actual processing and filtering
        self.filterAndProcess()

    '''
    This function will do the bulk of the processing for our tfidf matrix by filtering 
    our corpus to save time. The function will then calculate our actual tf and idf matrix
    but only store indices of tf values greater than 0 for our calculations
    '''
    def filterAndProcess(self):
        wordCount = 0
        docCount = len(self.fileids)
        mapWords = dict()
        for fileid in self.fileids:
            self.postProccesCorpus[fileid] = dict()

        # Filtering our corpus, saving a copy
        for fileid in self.fileids:
            for curWord in self.corpus.words(fileid):
                originalWord = curWord
                if originalWord in mapWords:
                    curWord = mapWords[originalWord]
                else:
                    if self.ignorecase == "yes":
                        curWord = curWord.lower()
                    # Doing our stemming work
                    curWord = self.stemmer.stem(curWord)
                    # Saving the word
                    mapWords[originalWord] = curWord
                # Checking our stopword file for stopwords
                if curWord in self.stopword:
                    continue
                # Adding current word to our processed corpus
                if curWord in self.postProccesCorpus[fileid]:
                    self.postProccesCorpus[fileid][curWord] += 1
                else:
                    self.postProccesCorpus[fileid][curWord] = 1
                # Checking if our current word has already been added and processed
                if curWord not in self.dictAllWords:
                    self.dictAllWords[curWord] = wordCount
                    self.allWords.append(curWord)
                    wordCount += 1

        # Processing for the tf
        # Storing actual locations of meaningful words (not 0)
        vectorNonZero = []
        for fileid in self.fileids:
            # Vector filled with 0s
            vectorTF = [0] * wordCount
            # This vector will hold location of non zero terms
            vectorNonZeroCurrent = []
            for curWord in self.postProccesCorpus[fileid].keys():
                x = self.postProccesCorpus[fileid][curWord]
                vectorTF[self.dictAllWords[curWord]] = x
                # Only storing meaningful terms location
                if x > 0:
                    vectorNonZeroCurrent.append(self.dictAllWords[curWord])
            vectorNonZero.append(vectorNonZeroCurrent)
            self.matrixTF.append(vectorTF)

        # Calcualte our actual IDF vector
        self.vectorIDF = [0] * wordCount
        for fileid in self.fileids:
            for curWord in self.postProccesCorpus[fileid].keys():
                self.vectorIDF[self.dictAllWords[curWord]] += 1

        # Calculate the actual TFIDF matrix
        for a in range(docCount):
            myVec = [0] * wordCount
            for j in vectorNonZero[a]:
                tempTF = self.matrixTF[a][j]
                tempIDF = self.vectorIDF[j]
                myVec[j] = self.logProcess(tempTF, tempIDF, docCount)
            self.tfidf.append(myVec)
        return

    '''
    This is our log function which will calculate the tfidf score
    according to user input
    '''
    def logProcess(self, tfRaw, idfRaw, docCount):
        # log normalized for tf
        if self.tf == "log" and tfRaw != 0:
            tfRaw = 1 + log(tfRaw, 2)
        # binary log for tf
        elif self.tf == "binary":
            tfRaw = 0 if tfRaw == 0 else 1
        # probabilistic inverse frequencey for idf
        if self.idf == "prob":
            if docCount == idfRaw:
                idfRaw = 0
            else:
                idfRaw = log((docCount - idfRaw) / float(idfRaw), 2)
        # inverse frequency smoothed for idf
        elif self.idf == "smooth":
            idfRaw = log(1 + docCount / float(idfRaw), 2)
        else:
            idfRaw = log(docCount / float(idfRaw), 2)
        return tfRaw * idfRaw

    '''
    the files of the corpus
    '''

    def fileids(self):
        return self.corpus.fileids()

    '''
    the raw content of the corpus
    '''

    def raw(self):
        return self.corpus.raw()

    '''
    the raw content of the specified files
    '''

    def raw(self, fileids):
        return self.corpus.raw(fileids)

    '''
    the words of the whole corpus
    '''

    def words(self):
        return self.corpus.allWords()

    '''
    the words of the specified fileids
    '''

    def words(self, fileids):
        return self.corpus.allWords(fileids)

    '''
    open a stream for reading the given corpus file
    '''
    def open(self, fileid):
        return self.corpus.open(fileid)

    '''
    the location of the given file on disk
    '''
    def abspath(self, fileid):
        return self.corpus.abspath(fileid)


    '''
    tf_idf():return a list of ALL tf-idf vector (each vector should be a list) for the corpus,
    ordered by the order where filelds are returned (the dimensions of the vector can be
    arbitrary, but need to be consistent with all vectors)

    tf_idf(fileid=fileid): return the tf-idf vector corresponding to that file

    tf_idf(filelist=[fileid]): return a list of vectors, corresponding to the tf-idf 
    to the list of fileid input
    '''
    def tf_idf(self, fileid=None, filelist=None):

        if fileid is not None:
            return self.tfidf[self.fileid.index(fileid)]
        elif filelist is not None:
            tempList = []
            for fId in filelist:
                tempList.append(self.tfidf[self.fileid.index(fileid)])
            return tempList
        return self.tfidf


    '''
    Return the list of the words corresponding to each vector of the tf-idf vector
    '''
    def tf_idf_dim(self):
        return self.allWords

    '''
    the input should be a list of words (treated as a document). The
    function should return a vector corresponding to the tf_idf vector for the new
    document (with the same stopword, stemming, ignorecase treatment applied, if
    necessary). You should use the idf for the original corpus to calculate the result (i.e. do
    not treat the document as a part of the original corpus)
    '''
    def tf_idf_new(self, words):
        wordCount = len(self.allWords)
        docCount = len(self.fileids)
        # Making our actual tf vector
        vectTF = [0] * wordCount
        # Using NLTK FreqDist to save time
        wordFreq = FreqDist(words)
        # Checking if we need to ignore case
        for curWord in wordFreq.keys():
            if self.ignorecase == "yes":
                curWord = curWord.lower()
            # Applying our porter stemmer
            curWord = self.stemmer.stem(curWord)
            # Checking if current word is a stop word
            if curWord in self.stopword:
                continue
            if curWord in self.dictAllWords:
                vectTF[self.dictAllWords[curWord]] = wordFreq[curWord]

        # Creating our actual tfidf vector
        finalVec = []
        for a in range(wordCount):
                tf = vectTF[a]
                idf = self.vectorIDF[a]
                finalVec.append(self.logProcess(tf, idf, docCount))
        return finalVec

    '''
    Return the cosine similarity between two documents in the corpus.
    '''
    def cosine_sim(self, fileid):
        x1 = self.tfidf[self.fileids.index(fileid[0])]
        x2 = self.tfidf[self.fileids.index(fileid[1])]
        return self.cosineCalc(x1, x2)

    '''
    the [words] is a list of words as is in the parameter of
    tf_idf_new() method. The fileid is the document in the corpus. The function return the
    cosine similarity between fileid and the new document specify by the [words] list. (Once
    again, use the idf of the original corpus). 
    '''
    def cosine_sim_new(self, words, fileid):
        x1 = self.tf_idf_new(words)
        x2 = self.tfidf[self.fileids.index(fileid)]
        return self.cosineCalc(x1, x2)

    '''
    breaking out the cosine similarity math into its own function 
    '''
    @staticmethod
    def cosineCalc(vecOne, vecTwo):
        sumOne = 0
        sumTwo = 0
        finalSum = 0

        for x in range(len(vecOne)):
            a = vecOne[x]
            b = vecTwo[x]
            sumOne += a * a
            sumTwo += b * b
            finalSum += a * b
        return finalSum / sqrt(sumOne * sumTwo)

