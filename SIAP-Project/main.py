from DataSetProcessing import *
from LinearRegressionProcessing import *
from TextProcessing import *

from LassoAndRidge import *
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
   
    dataSet1path = r'./assets/TreciIzvor.csv'
    dataSet2path = r'./assets/Comment_Sentiments.csv'

    commentsPath=r'./Comments2.csv'

    dataSet1 = LoadDataSet(dataSet1path)
    
    #comments = LoadDataSet(commentsPath)
    #comments = Preprocessing(comments)
    #comments_pp = GetSentiment(comments)
    #MergeComments(comments_pp)

    dataSet2 = LoadDataSet(dataSet2path)

    dataSet1 =  DropColumns(['Tags','Languages','Runtime','View Rating','Rotten Tomatoes Score','Awards Nominated For','Metacritic Score','Awards Received','Country Availability','Director','Writer','Actors','Boxoffice', 'Release Date', 'Netflix Release Date', 'Production House', 'Netflix Link', 'IMDb Link', 'Summary', 'Image', 'Poster', 'TMDb Trailer', 'Trailer Site'],dataSet1)
    
    dataSet1 = RemoveAllRowsExcept(dataSet1, 'Series or Movie', 'Series')
    dataSet1 = DropColumns(['Series or Movie'],dataSet1)

    dataSet1 = RemoveAllRowsWithNanField(dataSet1,'IMDb Score')
    dataSet1 = RemoveAllRowsWithNanField(dataSet1,'Hidden Gem Score')
    dataSet1 = RemoveAllRowsWithNanField(dataSet1,'IMDb Votes')

    dataSet1 = ToLowerStringColumnFields(dataSet1,'Title')
    #dataSet1 = ChangeNameOfColumn(dataSet1,"Title",'title')

    dataSet = MergeTwoDataSetsByColumn([dataSet1,dataSet2],['Title'])

    dataSet = DropColumns(['Unnamed: 0'],dataSet)
    dataSet = RemoveAllRowsWithNanField(dataSet,'Neg')
    dataSet = RemoveAllRowsWithNanField(dataSet,'Pos')
    dataSet = RemoveAllRowsWithNanField(dataSet,'Neu')
    dataSet = RemoveAllRowsWithNanField(dataSet,'IMDb Score')

    ExportDataSetAsCsv(dataSet,"ds3")


    ################################################
    # LASSO AND RIDGE

    ohe = OneHotEncoder()
    categorical_vars = ['Genre']
    encoded_data = ohe.fit_transform(dataSet[categorical_vars]).toarray()
    numerical_vars = ['Hidden Gem Score', 'Pos', 'Neg', 'Neu']
    processed_data = np.concatenate((encoded_data, dataSet[numerical_vars]), axis=1)

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(processed_data, dataSet['IMDb Score'], test_size=0.3, random_state=42)

    LassoModel(X_train, y_train, X_test, y_test)
    RidgeModel(X_train, y_train, X_test, y_test)

    ################################################

    # LINEAR REGRESSION

    #dataSet1 = MakeBinaryColumnsForColumn(dataSet1,'Genre')
    dataSet, newDataSet = SplitDataSet(dataSet1,['Title','Genre']) #new data set nosi imena filmova da kasnije napravimo lijep csv 

    x, y = SplitDataSet(dataSet,'IMDb Score')
   
    x_train, x_test,y_train,y_test = SplitDataSetTrainingAndTest(x, y, 0.2)

    y_pred = TrainAndPredict(x_train,y_train,x_test,y_test)

    #dataSet = ConcatDataSets([x_train,y_train,y_pred])
    



    #########################################################################

    #startIndex = 3100
    #ScrapeComments(dataSet1, startIndex)

