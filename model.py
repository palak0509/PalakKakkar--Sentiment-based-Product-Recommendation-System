## We need to have the 1 model for sentiment analysis and 1 recommendation system, for our analysis. 
## I've included the final processing part for the sentiment based recommendations, in the app.py notbook, itself, thus avoiding the need of an additional model.py file. 
## We can directly call the earlier exported pkl files directly to app.py notebook and use it for getting our final recommendations. 
## Thus we would not be using this (model.py) notebook for our analysis and recommendations.

# import libraries
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#--- HTML Tag Removal
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk



class Recommendation:

    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.data = pickle.load(open('rating.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_rating.pkl','rb'))
        self.model = pickle.load(open('logistic_model.pkl','rb'))
        self.raw_data = pd.read_csv("sample30.csv")
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)


    def getTopProducts(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Reviews'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)

    def getTopProductsNew(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Reviews'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)

    def getUsers(self):
        s= np.array(self.user_final_rating.index).tolist()
        #print(s)
        return ''.join(e+',' for e in s)


