###########################################################################
#Generic Calculations which include
#Standard Deviation Calculation,Mean Calculation
#Accuracy,Error Calculation
#Normalizing data by row and column
#Class Probability Calculation
#Predict class- argmax(class probabilities)
##########################################################################
from math import sqrt
from math import exp
from math import pi
import pandas as pd


def calcmean(datalist):
    return sum(datalist)/float(len(datalist))

def calcstd(datalist,flagnorm):
    smean= calcmean(datalist)
    if(flagnorm=="N"):
        var= sum([(x-smean)**2 for x in datalist]) / float(len(datalist)-1)
    else:
        var= sum([(x-smean)**2 for x in datalist]) / float(len(datalist))
        
    return sqrt(var)

def summary_dataset(dataframe,flagnorm="B"):
    dsummary=[]
    for col in dataframe.columns:
        dsummary.append([calcmean(dataframe[col]), calcstd(dataframe[col],flagnorm), len(dataframe[col])])
    return dsummary

def znormal(df):
    zdf=pd.DataFrame()
    tempdf=pd.DataFrame()
    summ=summary_dataset(df,"N")
    colcount=0
    for col in df.columns:
        tempdf=tempdf.assign(colnam=df[col].apply(lambda x:((x-summ[colcount][0])/summ[colcount][1])))
        zdf[col]=tempdf["colnam"]
        colcount+=1
    
    return zdf

def znormal_row(df):
    zdf=pd.DataFrame()
    tempdf=pd.DataFrame()
    df1=df.transpose()
    summ=summary_dataset(df1,"N")
    colcount=0
    for col in df1.columns:
        tempdf=tempdf.assign(colnam=df1[col].apply(lambda x:((x-summ[colcount][0])/summ[colcount][1])))
        zdf[col]=tempdf["colnam"]
        colcount+=1
    
    return zdf.transpose()


def calc_probguasdist(x,mu,sigma):
    epow=(-(x-mu)**2/(2*(sigma**2)))
    denom=sqrt(2*pi*(sigma**2))
    prob=float(exp(epow)/denom)
    return prob

def predict_class(x,summary,numclass):
    class_prob=list()
    for i in range(numclass):
        pclass=1/numclass
        pxclass=calc_probguasdist(x,summary[i][0],summary[i][1])
        class_prob.append(pclass*pxclass)
    
    pred_class="C"+str(class_prob.index(max(class_prob))+1)
    
    return pred_class

def predict_class_multiv(x,y,summ1,summ2,numclass):
    class_prob=list()
    for i in range(numclass):
        pclass=1/numclass
        pxclassv1=calc_probguasdist(x,summ1[i][0],summ1[i][1])
        pxclassv2=calc_probguasdist(y,summ2[i][0],summ2[i][1])
        class_prob.append(pclass*pxclassv1*pxclassv2)
    
    pred_class="C"+str(class_prob.index(max(class_prob))+1)
    
    return pred_class
        
def classifaccuracy(df):
    correct_prediction=[]
    countc1=df.query('PredictedValC1 == "C1"').PredictedValC1.count()
    countc2=df.query('PredictedValC2 == "C2"').PredictedValC2.count()
    countc3=df.query('PredictedValC3 == "C3"').PredictedValC3.count()
    countc4=df.query('PredictedValC4 == "C4"').PredictedValC4.count()
    countc5=df.query('PredictedValC5 == "C5"').PredictedValC5.count()
    
    columncount=df.count()
    totalpredics=sum(columncount[["PredictedValC1","PredictedValC2","PredictedValC3","PredictedValC4","PredictedValC5"]])
    classifacc=(countc1+countc2+countc3+countc4+countc5)/totalpredics
    correct_prediction.append(classifacc)
    correct_prediction.extend([[countc1,countc2,countc3,countc4,countc5]])
    return correct_prediction
    
def classiferror(df):
    wrong_prediction=[]
    countc1=df.query('PredictedValC1 != "C1"').PredictedValC1.count()
    countc2=df.query('PredictedValC2 != "C2"').PredictedValC2.count()
    countc3=df.query('PredictedValC3 != "C3"').PredictedValC3.count()
    countc4=df.query('PredictedValC4 != "C4"').PredictedValC4.count()
    countc5=df.query('PredictedValC5 != "C5"').PredictedValC5.count()
    
    columncount=df.count()
    totalpredics=sum(columncount[["PredictedValC1","PredictedValC2","PredictedValC3","PredictedValC4","PredictedValC5"]])
    classiferr=(countc1+countc2+countc3+countc4+countc5)/totalpredics
    wrong_prediction.append(classiferr)
    wrong_prediction.extend([[countc1,countc2,countc3,countc4,countc5]])
    return wrong_prediction


