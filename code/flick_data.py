

import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

num_v = 1752
num_u = 924

def tf(word, blob):
    return float(blob.words.count(word)) / float(len(blob.words))

def n_containing(word, bloblist):
    count = 0.0
    # print type(bloblist[0])
    for blob in bloblist:
        if word in blob.words:
            count+=1

    return float(count)

def idf(word, bloblist):
    return math.log(float(len(bloblist)) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def movie_feature_tf(description):

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
    # for word, score in sorted_words[:800]:
    #     print("\tWord: {}, TF-IDF: {}".format(word, score))
    num_words = 4000
    # print sorted_words[:num_words]
    vocab = {}
    for i in range(num_words):
        vocab[sorted_words[i][0]] = i

    f = open("mult_flkr.txt",'w')

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

def movie_feature(description):
    stop = set(stopwords.words('english'))
    words = {}

    for des in description:
        # print des
        # temp = word_tokenize(des)
        tokenizer = RegexpTokenizer(r'\w+')
        temp = tokenizer.tokenize(des)
        # print temp
        for i in temp:
            if (i not in stop):
                if (i in words.keys()):
                    words[i] += 1
                else:
                    words[i] = 1

    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    num_words = 4000
    # print sorted_words[:num_words]
    vocab = {}
    for i in range(num_words):
        vocab[sorted_words[i][0]] = i

    f = open("mult_flkr.txt",'w')

    for des in range(len(description)):
        feature = {}
        tokenizer = RegexpTokenizer(r'\w+')
        temp = tokenizer.tokenize(description[des])
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


    # print len(words)


movies = []
description = ["" for i in range(num_v)]

with open('Dataset/FlickscoreData-26oct2018/movies.json') as fp:
    line = fp.readline()
    # print line
    count = 0
    while line:
        temp = json.loads(line.strip())
        if (temp['description'] != ""):
            movies.append(temp)
            description[count] = tb(temp['description']) 
            # description[count] = temp['description']


        # print temp['genre']
        # for des in (temp['description']):
            # description[count] = tb(des)
        # description[count] = tb(temp['description'])              


            # print count
            count +=1
        line = fp.readline()

user = []

with open('Dataset/FlickscoreData-26oct2018/users.json') as fp:
    line = fp.readline()
    # print line
    count = 0
    while line:
        temp = json.loads(line.strip())
        user.append(temp) 
        # print count
        count +=1
        line = fp.readline()


print "number of movies - " 
print len(movies)
print "number of users - " 
print len(user)

movie_id ={}
movie_name ={}
for m in range(len(movies)):
    movie_id[movies[m][u'movie_id']] = m
    movie_name[m] = movies[m][u'name']
# print movie_id


user_id ={}
for u in range(len(user)):
    user_id[user[u][u'_id']] = u
# print user_id
# return

with open('Dataset/FlickscoreData-26oct2018/ratings_correctFormat.json') as fp:
    ratings = json.load(fp)
R_train = np.mat(np.zeros((num_u,num_v)))
R_test = np.mat(np.zeros((num_u,num_v)))
# for e in range(1,6):
e = 1
save_file1 = "user" + str(e) + "_train.txt"
save_file2 = "user" + str(e) + "_test.txt"
# with open(save_file1,'w') as fp1, open(save_file2,'w') as fp2 :
list1 = []
c = 0
for i in range(len(ratings)):
    keys = ratings[i]['rated'].keys()
    ind = np.random.choice(len(keys), int(0.8*len(keys)))
#         print len(keys)
    if (len(ind) < 20):
        ind = [w for w in range(len(keys))]
    for j in range(len(keys)):
        if (j in ind):
            if (keys[j] != u'submit'):
                if (keys[j] in movie_id.keys() and int(ratings[i]['rated'][keys[j]][0]) > 0):
                    R_train[user_id[ratings[i]['_id']],movie_id[keys[j]]] = 1
                    c += 1
                # elif (keys[j] in movie_id.keys() and int(ratings[i]['rated'][keys[j]][0]) < 0):
                #     R_train[user_id[ratings[i]['_id']],movie_id[keys[j]]] = -1
                # fp1.write(str(user_id[ratings[i]['_id']]) + '\t' + str(movie_id[keys[j]]) + '\t' + ratings[i]['rated'][keys[j]][0] + '\n')
        else:
            if (keys[j] != u'submit'):
                if (keys[j] in movie_id.keys() and int(ratings[i]['rated'][keys[j]][0]) > 0):
                    R_test[user_id[ratings[i]['_id']],movie_id[keys[j]]] = 1
                if (keys[j] in movie_id.keys()):
                    list1.append((user_id[ratings[i]['_id']],movie_id[keys[j]]))

print len(list1)
print c
with open("flk_R_train.pkl","w") as f:
    pickle.dump(R_train,f)

with open("flk_R_test.pkl","w") as f:
    pickle.dump((R_test, list1),f)

with open("flk_movie_name.pkl","w") as f:
    pickle.dump(movie_name,f)

# movie_feature(description)
movie_feature_tf(description)



















                    

