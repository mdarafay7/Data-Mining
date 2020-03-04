import os
import nltk
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import timeit




start = timeit.default_timer()

def normalize(myDict):
    sum = 0
    for i in myDict:
        sum += myDict[i] ** 2

    normalizer=math.sqrt(sum)

    for i in myDict:
        myDict[i]=myDict[i]/normalizer

    return myDict


nltk.download('stopwords')
corpusroot = './presidential_debates/presidential_debates'
debates_store={}
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    debates_store.update({filename:doc})


debates_tokenized_store={}
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
for debate in debates_store:
    tokens=tokenizer.tokenize(debates_store[debate])
    debates_tokenized_store.update({debate : tokens})


debates_stopwords_removed={}
stop_words = set(stopwords.words('english'))
for debate in debates_tokenized_store:
    filtered_debate=[]
    for word in debates_tokenized_store[debate]:
        if word not in stop_words:
            filtered_debate.append(word)
    debates_stopwords_removed.update({debate : filtered_debate})


debates_stemmed={}
stemmer = PorterStemmer()
for debate in debates_stopwords_removed:
    stemmed_debate=[]
    for word in debates_stopwords_removed[debate]:
        stemmed_debate.append(stemmer.stem(word))
    debates_stemmed.update({debate : stemmed_debate})

word_freq_dict={}
counter=0
tf_idf_vector_store={}
document_frequency_vector={}
#
for debate in debates_stemmed:
    tf_idf_vector={}
    for word in debates_stemmed[debate]:
        if word in tf_idf_vector:
            continue
        tf=debates_stemmed[debate].count(word)
        df=0
        if tf:
            if word in document_frequency_vector:
                df=document_frequency_vector.get(word)
            else:
                for search_debate in debates_stemmed:
                    if word in debates_stemmed[search_debate]:
                        df+=1
                document_frequency_vector.update({word : df})
        else:
            if word in document_frequency_vector:
                df=document_frequency_vector.get(word)
            else:
                for search_debate in debates_stemmed:
                    if word in debates_stemmed[search_debate]:
                        df+=1
                document_frequency_vector.update({word : df})

        tf_idf=(1+math.log(tf,10))*(math.log(((30)/(df)),10))
        tf_idf_vector.update({word : tf_idf})

    tf_idf_vector_store.update({debate : normalize(tf_idf_vector)})


# print(tf_idf_vector_store)
stop = timeit.default_timer()
print('Time: ', stop - start)


print(tf_idf_vector_store["1976-10-22.txt"][stemmer.stem("agenda")])
