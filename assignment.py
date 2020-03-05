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
#read corpus
corpusroot = './presidential_debates/presidential_debates'
debates_store={}
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    debates_store.update({filename:doc})

#tokenize
debates_tokenized_store={}
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
for debate in debates_store:
    tokens=tokenizer.tokenize(debates_store[debate])
    debates_tokenized_store.update({debate : tokens})

#stopwords remove
debates_stopwords_removed={}
stop_words = set(stopwords.words('english'))
for debate in debates_tokenized_store:
    filtered_debate=[]
    for word in debates_tokenized_store[debate]:
        if word not in stop_words:
            filtered_debate.append(word)
    debates_stopwords_removed.update({debate : filtered_debate})

#stem
debates_stemmed={}
stemmer = PorterStemmer()
for debate in debates_stopwords_removed:
    stemmed_debate=[]
    for word in debates_stopwords_removed[debate]:
        stemmed_debate.append(stemmer.stem(word))
    debates_stemmed.update({debate : stemmed_debate})

#tf-idf
word_freq_dict={}
tf_idf_vector_store={}
document_frequency_vector={}
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

# postings list
postings_list_store={}
for debate in tf_idf_vector_store:
    for word in tf_idf_vector_store[debate]:
        postings_list={}
        postings_list_sorted={}
        if tf_idf_vector_store[debate][word] in postings_list_store:
            continue
        else:
            for search_debate in tf_idf_vector_store:
                if word in tf_idf_vector_store[search_debate]:
                    postings_list.update({search_debate : tf_idf_vector_store[search_debate][word]})

                else:
                    continue

        order=sorted(postings_list, key=postings_list.get, reverse=True)[:10]
        for val in order:
            postings_list_sorted.update({val : postings_list[val]})

        postings_list_store.update({word : postings_list_sorted})

#
# print(postings_list_store[stemmer.stem("clinton")])



def getweight(document,token):
    return(tf_idf_vector_store[document][token])

def getidf(token):
    df=0
    if token in document_frequency_vector:
        df=document_frequency_vector.get(token)
    else:
        for search_debate in debates_stemmed:
            if word in debates_stemmed[search_debate]:
                df+=1

    return(math.log((30/df),10))

def query(query):
    store={}
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    query = query.lower()
    query=tokenizer.tokenize(query)

    tf_idf_vector_store={}
    for token in query:
        tf_idf=1+math.log(query.count(token),10)
        tf_idf_vector_store.update({token : tf_idf})

    tf_idf_vector_store=normalize(tf_idf_vector_store)


    for document in debates_store:
        summer=[]
        for token in query:
            if stemmer.stem(token) not in postings_list_store:
                continue
            if document in postings_list_store[stemmer.stem(token)]:
                query_vector=tf_idf_vector_store[token]
                summer.append((query_vector*postings_list_store[stemmer.stem(token)][document]))
            else:
                query_vector=tf_idf_vector_store[token]
                upper_bound=postings_list_store[stemmer.stem(token)][list(postings_list_store[stemmer.stem(token)])[-1]]
                summer.append(query_vector*(upper_bound))
        store.update({document : sum(summer)})

    order=sorted(store, key=store.get, reverse=True)[:1]

    return store[order[0]]


print(getweight("2012-10-03.txt","health"))
print(query("health insurance wall street"))
print(getweight("2012-10-03.txt","health"))
print(getidf("health"))


stop = timeit.default_timer()
print('Time: ', stop - start)
