import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import csv

num_v = 285
num_u = 1226

def tf(word, blob):
    return float(blob.words.count(word)) / len(blob.words)

def n_containing(word, bloblist):
    count = 0.0
    # print type(bloblist[0])
    for blob in bloblist:
        if word in blob.words:
            count+=1

    return count

def idf(word, bloblist):
    return math.log(float(len(bloblist)) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def artist_feature(description):

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())
    # print len(vectorizer.get_feature_names())
    stop = set(stopwords.words('english'))
    # print type(description[0])
    scores = {}
    for i, blob in enumerate(description):
        print("Top words in document {}".format(i + 1))
        for word in blob.words:
            if word not in stop:
                temp = tfidf(word, blob, description)
                if word in scores:
                    if (temp > scores[word]): 
                        scores[word] = temp
                else:
                    scores[word] = temp 
    # scores = {word: tfidf(word, blob, bloblist) for word in blob.words if word not in stop}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print len(sorted_words)
    num_words = 4000
    for word, score in sorted_words[:num_words]:
        print("\tWord: {}, TF-IDF: {}".format(word.encode('utf-8'), score))
    vocab = {}
    for i in range(num_words):
        vocab[sorted_words[i][0]] = i
    f = open("mult_music.txt",'w')

    for des in range(len(description)):
        feature = {}
        tokenizer = RegexpTokenizer(r'\w+')
        temp = description[des].words
        for i in temp:
            if (i not in stop and i in vocab.keys()):
                if (vocab[i] in feature.keys()):
                    feature[vocab[i]] += 1
                else:
                    feature[vocab[i]] = 1
        str1 = str(des)
        for i in feature.keys():
            str1 += " " + str(i) + ":" + str(feature[i])
        f.write(str1 + "\n")

    f.close()


def train_test_split():
	R_train = np.mat(np.zeros((num_u,num_v)))
	R_test = np.mat(np.zeros((num_u,num_v)))
	list1 = []
	with open('Dataset/Last_fm_dataset/lastfm.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader)
		count = 0	
		for row in csv_reader:
			# print row
			indices = [i for i, x in enumerate(row[1:]) if int(x) == 1]
			# print len(indices)
			if (len(indices) > 10):
				ind = np.random.choice(len(indices), int(0.4*len(indices)))
			elif (len(indices) > 0):
				ind = np.random.choice(len(indices),1)
			f_ind = [indices[i] for i in ind]
			print len(ind)
			for i in indices:
				if i in f_ind:
					R_train[count,i] = 1
				else:
					R_test[count,i] = 1	
					# list1.append((count,))
			count += 1

	# print R_train[0,:]
	# return		
	with open("Fm_R_train.pkl","w") as f:
		pickle.dump(R_train,f)

	with open("Fm_R_test.pkl","w") as f:
		pickle.dump(R_test,f)

# def artist_feature(description):
#     stop = set(stopwords.words('english'))
#     words = {}

#     for des in description:
#         # print des
#         # temp = word_tokenize(des)
#         tokenizer = RegexpTokenizer(r'\w+')
#         temp = tokenizer.tokenize(description[des])
#         # print temp
#         for i in temp:
#             if (i not in stop):
#                 if (i in words.keys()):
#                     words[i] += 1
#                 else:
#                     words[i] = 1

#     sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
#     num_words = 4000
#     # print sorted_words[:num_words]
#     vocab = {}
#     for i in range(num_words):
#         vocab[sorted_words[i][0]] = i

#     f = open("mult_music.txt",'w')

#     for des in range(len(description)):
#         feature = {}
#         tokenizer = RegexpTokenizer(r'\w+')
#         temp = tokenizer.tokenize(description[des])
#         for i in temp:
#             if (i not in stop and i in vocab.keys()):
#                 if (vocab[i] in feature.keys()):
#                     feature[vocab[i]] += 1
#                 else:
#                     feature[vocab[i]] = 1
#         str1 = str(des)
#         for i in feature.keys():
#             str1 += " " + str(i) + ":" + str(feature[i])
#         f.write(str1 + "\n")

#     f.close()

art_info = {}
with open('Dataset/Last_fm_dataset/art_info.json', 'rb') as outfile:
    art_info = json.load(outfile)

header = []
artist_name ={}
with open('Dataset/Last_fm_dataset/lastfm.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	header = next(csv_reader)
header = header[1:]
for i in range(len(header)):
	artist_name[i] = header[i]

description = []
for i in art_info:
	description.append(tb(art_info[i]))
artist_feature(description)
train_test_split()
with open("artist_name.pkl","w") as f:
    pickle.dump(artist_name,f)



