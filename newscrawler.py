# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 19:40:39 2014

@author: Administrator
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import urllib2,re,sys,json,os,string,gensim,nltk
from nltk import *
from nltk.corpus import gutenberg
from bs4 import BeautifulSoup
from urllib import urlencode
from nltk.probability import *
import numpy as np  
import matplotlib.pyplot as plt 
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
def geturl(begin_url):
    """get url"""
    encoding='utf-8'
    content=urllib2.urlopen(begin_url).read() 
    ss=content.replace(" ","")
    urls=re.findall(r"href=.*?content.*?.htm",ss,re.I)
    re_r=re.compile('href="+')
    urls = [re_r.sub('',str(url)) for url in urls]
    return urls
def getnews(url,news):
    """"get the content of news"""
    content=urllib2.urlopen(url).read() 
    soup = BeautifulSoup(content)
    title = soup.title
    re_h=re.compile('</?\w+[^>]*>')
    content =soup.find_all('p')
    content=re_h.sub('',str(content))
    title=re_h.sub('',str(title))
    doc = title + content
    news.append(doc)
    return news
def dealwithnews(news):
    """deal with the news that are crawled from website,
    and output some relative information""" 
    text = nltk.word_tokenize(news)
    #print text
    print "The number of words in the news is:"
    print len(text)
    print "The number of words which include uppercase letter is:"
    print len([item for item in text if re.match('[A-Z]',item)])
    #print [item for item in text if re.match('[A-Z]',item)]
    print "The number of vbers in the news is:"
    #print nltk.pos_tag(text)
    vbers = [item[0] for item in nltk.pos_tag(text) if item[1]=='VB']
    print len(vbers)
    print "The number of nouns in the news is:"
    nouns = [item[0] for item in nltk.pos_tag(text) if item[1]=='NN']
    print len(nouns)
    print "The number of sentences in the news is:"
    sentences = news.split('.')#Rough estimate,  need to improve
    print len(sentences)
    text = nltk.Text(word.lower() for word in text)
    print "the negative word in the news are:"
    text.similar('sad')
    print "the positive word in the news are:"
    text.similar('happy')
"""-----------------------------------------Statistics--------------------------------------------"""
news = []
begin_url='http://www.chinadaily.com.cn/'
urls = geturl(begin_url)
i=0
while i<10:
    url="http://www.chinadaily.com.cn/" + str(urls[i])
    print url
    news = getnews(url,news)
    i = i+1
i=1
for new in news:
    print "The " + str(i)+ " th news's condition is as follow:"
    print '\n'
    dealwithnews(new)
    print '\n'
    i=i+1
"""------------------------------------------Visualization-----------------------------------------"""
'''build ldamodel for the news'''
text = [nltk.word_tokenize(new) for new in news]
text = [[w.lower()for w in item if len(w)>3]for item in text]
stopwords = nltk.corpus.stopwords.words('english')
text = [[w for w in item if w not in stopwords]for item in text]
dictionary = corpora.Dictionary(text)
corpus = [dictionary.doc2bow(document) for document in text]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = gensim.models.ldamodel.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=10,update_every=0,passes=10)
index = similarities.MatrixSimilarity(lda[corpus])
string = lda.show_topics(topn=10, formatted=False)
words = []
for w in string:
   words.append([item[1] for item in w])
'''calculate the words' frenquency'''
content = []
for item in text:
    content+=item
corpus = dictionary.doc2bow(content)
wordlist = sorted(corpus, key=lambda corpus : corpus[1],reverse=True)
wordlist = wordlist[0:20]
#print wordlist
keylist=[dictionary[item[0]]for item in wordlist]
#print keylist
vallist=[item[-1]for item in wordlist]
#print vallist
'''search the word belong to which topic'''
'''colorlist = []
for w in keylist:
    i=0
    while i<10:
        if w in words[i]:
            colorlist+=[i]
            continue
        i=i+1
    if i==10:
        colorlist+=[10]
print colorlist'''   
barwidth=0.2
xVal=numpy.arange(len(keylist))
plt.xticks(xVal+barwidth/2.0,keylist,rotation=30)
plt.bar(xVal,vallist,width=barwidth,color='y')
plt.title(u'word frequency distribution histogram')
plt.show()
'''------------------------------------------end---------------------------------'''