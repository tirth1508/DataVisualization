# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:45:22 2021

@author: Nikhi
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from nltk import bigrams
from nltk import FreqDist
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

data = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

#print(data.columns)
print(data.info())

# 100 * data.isnull().sum()/len(data)

### DATA CLEANING

null_df = round(data.isnull().mean()*100,2)
null_df.sort_values(inplace=True)
null_df = null_df[null_df > 0]

fig = plt.figure(figsize = (20,20))
sns.set(style = 'whitegrid')
ax = sns.barplot(x  = null_df.index.tolist(), y = null_df, palette = 'hot_r')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.show()

del data["Flight #"]
del data["Route"]
del data["cn/In"]
del data["Registration"]
del data["Ground"]

print(data.Time.dtype)

for i in data.Time.dropna():
    try:
        pd.to_datetime(i)
    except:
        print(i)

#data["Date"] = pd.to_datetime(data["Date"])

data.Time=data.Time.replace({"c: 1:00":'1:00', "c:17:00":'17:00', "c: 2:00":'2:00', "c:09:00":'09:00',
                          "c16:50":'16:50', "12'20":'12:20', "18.40":'18:40', "c:09:00":'09:00',
                          "114:20":'14:20', "c14:30":'14:30', "0943":'09:43', "22'08":'22:08', "c: 9:40":'9:40'})

data['Time'] = data['Time'].replace(np.nan, '00:00') 
data['Time'] = data['Time'].str.replace('c: ', '')
data['Time'] = data['Time'].str.replace('c:', '')
data['Time'] = data['Time'].str.replace('c', '')
data['Time'] = data['Time'].str.replace('12\'20', '12:20')
data['Time'] = data['Time'].str.replace('18.40', '18:40')
data['Time'] = data['Time'].str.replace('0943', '09:43')
data['Time'] = data['Time'].str.replace('22\'08', '22:08')
data['Time'] = data['Time'].str.replace('114:20', '00:00')

data['Time'] = data['Date'] + ' ' + data['Time']
def todate(x):
    return datetime.strptime(x, '%m/%d/%Y %H:%M')
data['Time'] = data['Time'].apply(todate)

sns.barplot(data.groupby(data.Time.dt.month)[['Date']].count().index, 'Date', data=data.groupby(data.Time.dt.month)[['Date']].count(), color='lightgreen', linewidth=2)
plt.xticks(data.groupby(data.Time.dt.month)[['Date']].count().index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


loc=[]
def get_country(location):
    if type(location)==float:
        return "Unknown"
    
    loc=location.split(',')
    return loc[len(loc)-1].replace(" ",'').lower().replace("off ",'')

data.Location=data.Location.apply(get_country)

len(data.Operator.unique())


### MILITARY vs PASSENGERS

data.Operator = data.Operator.str.upper()
data['isMilitary'] = data.Operator.str.contains('MILITARY')

count = data.groupby('isMilitary')[['isMilitary']].count()
count.index = ['Passenger', 'Military']

colors = ['skyblue', 'maroon']
patches, texts = plt.pie(count.isMilitary, colors=colors, labels=count.isMilitary, startangle=90)
plt.legend(patches, count.index, loc="best", fontsize=10)
plt.title('Total number of accidents by Type of flight', loc='Center', fontsize=14)


### TRENDS OF SURVIVAL RATES

data["Date"] = pd.to_datetime(data["Date"])
data["Survival_Rate"] = 100 * (data["Aboard"] - data["Fatalities"]) / data["Aboard"]
yearly_survival = data[["Date","Survival_Rate"]].groupby(data["Date"].dt.year).agg(["mean"])

yearly_survival.plot(legend=None)


### FATALITIES BY EACH OPERATOR

data.Operator = data.Operator.str.upper()
#data.Operator = data.Operator.replace('A B AEROTRANSPORT', 'AB AEROTRANSPORT')

Prop_by_Op = data.groupby('Operator')[['Fatalities']].sum()
Prop_by_Op = Prop_by_Op.rename(columns={"Operator": "Fatalities"})
Prop_by_Op = Prop_by_Op.sort_values(by='Fatalities', ascending=False)
Prop_by_OpTOP = Prop_by_Op.head(15)

sns.barplot(y=Prop_by_OpTOP.index, x="Fatalities", data=Prop_by_OpTOP, palette="gist_heat", orient='h')

summary = data["Summary"].dropna()
summary = pd.DataFrame(summary)

summ_df = summary["Summary"].tolist()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summ_df)

random_state=123
model = MiniBatchKMeans(n_clusters=6, random_state=random_state)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()

for i in range(6):
    print('Cluster %d:' % i)
    for ind in order_centroids[i, :30]:
        print ('%s' % terms[ind]),  
    print("\n")

def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join([i for i in s if i not in exclude])
    return s

book = data['Summary'].str.lower().dropna().apply(remove_punctuation).str.split().values.sum()
#book = remove_punctuation(book)
stop = stopwords.words('english')
wrd = [w for w in book if w not in stop]

bigram_df = list(bigrams(wrd))
fdistBigram = FreqDist(bigram_df)
fdistBigram.plot(20)

wordcloud = WordCloud(background_color="white",random_state = 2016).generate(" ".join([i for i in terms[i,:10]]))
plt.imshow(wordcloud)