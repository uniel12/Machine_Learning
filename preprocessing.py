#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 등장 빈도 기준 정제 함수 정의

def clean_by_freq(tokenized_words, freq):
    from collections import Counter
    # 1. Counter 함수를 통해 단어의 빈도수를 카운트하여 단어 집합 생성
    # Vocab 이라는 변수에 담기
    vocab = Counter(tokenized_words)
    # 2. 빈도수가 freq 이하인 단어 추출
    # low_freq_words 라는 변수에 담기
    low_freq_words= []
    for key, value in vocab.items():
        if value <= freq:
            low_freq_words.append(key)
    # 3. low_freq_words에 포함되지 않는 단어 리스트 생성
    # cleaned_words 라는 변수에 담기
    cleaned_words= []
    for word in tokenized_words:
        if word not in low_freq_words:
            cleaned_words.append(word)
    return cleaned_words


# In[2]:


# 단어 길이 기준 정제함수
def clean_by_len(tokenized_words, length):
    cleaned_by_freq_len = []
    # 단어 길이가 length 이상인 단어들을 cleaned_by_freq_len 이라는 변수에 담기
    for word in tokenized_words:
        if len(word) >= length:
            cleaned_by_freq_len.append(word)
    return cleaned_by_freq_len


# In[3]:


# 불용어 제거 함수 만들기
def clean_by_stopwords(tokenized_words,stopwords_set):
    cleaned_words = []
    
    # 불용어 제거하는 코드 작성
    for word in tokenized_words:
        if word not in stopwords_set:
             cleaned_words.append(word)
                
    return cleaned_words


# In[4]:


from nltk.stem import PorterStemmer

# 포터스테머 어간 추출 함수
def stemming_by_porter(tokenized_words):
    porter_stemmer = PorterStemmer()
    porter_stemmed_words =[]
    for word in tokenized_words :
        stem = porter_stemmer.stem(word)
        porter_stemmed_words.append(stem)
    return porter_stemmed_words


# In[5]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

# 품사 태깅 함수 정의
def pos_tagger(tokenized_sents):
    pos_tagged_words = []
    for sentence in tokenized_sents:
        # 단어 토큰화
        tokenized_words = word_tokenize(sentence)
        
        # 품사 태깅
        pos_tagged = pos_tag(tokenized_words)
        
        # 품사태깅한 데이터 담아주기 -extend 활용
        pos_tagged_words.extend(pos_tagged)
    return pos_tagged_words


# In[6]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

# pennTree -> WordNet으로 변환
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    else:
        return
    
# 표제어 추출해주는 함수 정의
def word_lemmatizer(pos_tagged_words):
    lemmatizer = WordNetLemmatizer() # 객체 생성
    lemmatized_words = [] # 표제어 추출된 단어를 담는 리스트
    
    for word, tag in pos_tagged_words:
        wn_tag = penn_to_wn(tag)
        
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag)) # 표제어 추출 함수
        else :
            lemmatized_words.append(word)           
    return lemmatized_words


# In[ ]:





# In[ ]:




