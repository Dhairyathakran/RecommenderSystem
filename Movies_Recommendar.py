# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:07:35 2023

@author: dhair
"""
#****************  Libraries  *************

import numpy as np
import pandas as pd 
import ast
import nltk 
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
#****************** DataSet ***************

movies = pd.read_csv('/Users/dhair/OneDrive/Desktop/tmdb_5000_movies.csv')

credit = pd.read_csv('/Users/dhair/OneDrive/Desktop/tmdb_5000_credits.csv')

print(movies)
print(credit)

#********************* Merge The Datasets ****************

movies = movies.merge(credit , on = 'title')
print(movies.shape)
print(movies)

#*********** DropOut the columns which are not useful ******************

movies = movies[['id' ,'keywords', 'title' , 'overview' , 'genres' , 'cast' , 'crew']]
print(movies)

#********* Check null values inside the data set ****************

print(movies.isnull().sum()) 
print(movies.dropna(inplace = True))

#******* Check Duplicates inside the Dataset************* 

print(movies.duplicated().sum)
#print(movies)

#print(movies.iloc[0].genres)

#*************** Create a function ************

#a = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]

'''def convert(obj):
    l = []
    for i in obj:
        l.append(i['name'])
    return l
        

print(ast.literal_eval([{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]))
'''

def convert (obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
        

    
movies['genres'] = movies['genres'].apply(convert)  
#print(movies['genres'])
 
movies['keywords'] = movies['keywords'].apply(convert)
#print(movies['keywords'])

#************ creating another funtion ***********

#print(movies.iloc[0].keywords)

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            L.append(i['name'])
            count += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
#print(movies['cast'])

#************* Create another Function to fetch the directorname ************

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
#print(movies['crew'])        
        
movies['overview'] = movies['overview'].apply(lambda x:x.split())
#print(movies['overview'])

#***********Transformation Remove the space between words ****************

movies['genres']   = movies['genres'].apply(lambda x:[i.replace(" " , "")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" " , "")for i in x])
movies['cast']     = movies['cast'].apply(lambda x:[i.replace(" " , "")for i in x])
movies['crew']     = movies['crew'].apply(lambda x:[i.replace(" " , "")for i in x])

#*********** Concatenate the columns in one column ******************

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

#****************** Creating a new DataFrame *******************

new_df = movies[['id' , 'title' , 'tags']]
#print(new_df)

#**************** Covert List into the Strings *********************

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
#print(new_df['tags'])

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
#print(new_df['tags'][0])

#********** Do Stemming Here *************


ps = PorterStemmer()

def stem (text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)
#print(new_df['tags'])

#********* Apply CountVectorizer Algorithm on text ***************


cv = CountVectorizer(max_features = 5000 , stop_words = 'english')

vectors = cv.fit_transform(new_df['tags']).toarray()
print (vectors.shape)

#print(cv.get_feature_names())

#******* Calculate the cosine Distance between vectors **************


similarity = cosine_similarity(vectors)
print(sorted(list(enumerate(similarity[0])) , reverse = True , key = lambda x:x[1])[1:6]) 

#********** Create a function find the similarity *****************

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)) , reverse = True , key = lambda x:x[1])[1:6]
    
    
    for i in movie_list :
        print(new_df.iloc[i[0]].title)
        
print('Similar Movies NAmes : ', recommend('Avatar'))

#******* Import Pickel LIbrary *********


pickle.dump(new_df,open('movie.pkl' , 'wb'))

#********* Create The dictnory reather then the DataFrame ******

pickle.dump(new_df.to_dict,open('movie_dict.pkl' , 'wb'))

#*********** Dump the similarity *************
pickle.dump(similarity,open('similarity.pkl' , 'wb'))







