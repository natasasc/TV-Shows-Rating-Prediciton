from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from nltk.corpus import stopwords
from textblob import TextBlob, Word
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def Preprocessing(ds):
    for i in range(len(ds)):
        ds.apply(lambda x: x.astype(str).str.lower())
    stop=stopwords.words('english')
    ds=ds.apply(lambda x: x.astype(str).str.lower())
    #removing punctuations
    ds['Comment']=ds['Comment'].str.replace('[^\w\s]','')
    #removing stopwords
    ds['Comment']=ds['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #stemming words
    ds['Comment']=ds['Comment'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    ds.to_csv('comments_pp.csv')
    return ds

def GetSentiment(ds):
    newDs=ds
    sentiments=[]
    
    # nltk.download('vader_lexicon')
    
    for c in ds['Comment']:
        
        analysis = TextBlob(c)
        score = SentimentIntensityAnalyzer().polarity_scores(c)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        
        if (neu > pos and neu> neg) or pos==neg:
           sentiments.append(2)
        elif pos>neg:
            sentiments.append(1)
        elif neg > pos :
            sentiments.append(0)
       
    newDs.insert(loc=2,column='sentiment',value=sentiments)
    newDs.to_csv('comments_pp.csv')
    return newDs
            
            
def MergeComments(ds):
    title='!@#$%^&^%$#@'
    nds=pd.DataFrame(columns=["Title","Neg","Pos","Neu"])
    titles=[]
    values=[]
    pos=0
    neg=0
    neu=0
    cnt=0
    for index,obj in ds.iterrows():
        if obj['Title']!=title:
           nds.loc[cnt]=[title,neg,pos,neu]
           cnt+=1
           title=obj['Title']
           neg=0
           pos=0
           neu=0
        if obj['sentiment']==0:
            neg+=1
        elif obj['sentiment']==1:
            pos+=1
        elif obj['sentiment']==2:
            neu+=1
            
    nds.to_csv('Comment_Sentiments.csv')
            