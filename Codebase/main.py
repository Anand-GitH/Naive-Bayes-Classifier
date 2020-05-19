import pandas as pd
from genericcalc import summary_dataset
from genericcalc import predict_class
from genericcalc import classifaccuracy
from genericcalc import classiferror
from genericcalc import znormal
from genericcalc import predict_class_multiv

def main():
    df1 = pd.read_excel (r'F1Data.xlsx')
    df2 = pd.read_excel (r'F2Data.xlsx')
 
##############We have 5 classes so all are equiprobable###################
    pc1=pc2=pc3=pc4=pc5=1/5
    
##########################################################################
#Case1: Training with first 100 data points 
#Calculate mean and standard deviation for first 100 data points
#Use it to calculate probability of p(F1|Ci)
#########################################################################
    summ=summary_dataset(df1[0:100])
    predict_data=df1[100:]
    predict_data=predict_data.assign(PredictedValC1=predict_data["Var1"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC2=predict_data["Var2"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC3=predict_data["Var3"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC4=predict_data["Var4"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC5=predict_data["Var5"].apply(lambda x:predict_class(x,summ,5)))
    accuracy=round(classifaccuracy(predict_data),2)
    errrate=round(classiferror(predict_data),2)
    print('Case 1: X=F1 Accuracy:',accuracy,' and Error:',errrate)
    
##########################################################################
#Case2: Training with first 100 data points on normalized F1
#Calculate mean and standard deviation for first 100 data points which will 
#be zero and sd will be 1 as data is normalized
#Use it to calculate probability of p(Z1|Ci)
#########################################################################
    normdf=znormal(df1)
    summ=summary_dataset(normdf[0:100])
    predict_data=df1[100:]
    predict_data=predict_data.assign(PredictedValC1=predict_data["Var1"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC2=predict_data["Var2"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC3=predict_data["Var3"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC4=predict_data["Var4"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC5=predict_data["Var5"].apply(lambda x:predict_class(x,summ,5)))
    accuracy=round(classifaccuracy(predict_data),2)
    errrate=round(classiferror(predict_data),2)
    print('Case 2: X=Z1 Accuracy:',accuracy,' and Error:',errrate)
    
    
##########################################################################
#Case3: Training with first 100 data points using F2
#Calculate mean and standard deviation for first 100 data points of F2
#Use it to calculate probability of p(F1|Ci) but using F2 mean and std
#########################################################################    
    summ=summary_dataset(df2[0:100])
    predict_data=df1[100:]
    predict_data=predict_data.assign(PredictedValC1=predict_data["Var1"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC2=predict_data["Var2"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC3=predict_data["Var3"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC4=predict_data["Var4"].apply(lambda x:predict_class(x,summ,5)))
    predict_data=predict_data.assign(PredictedValC5=predict_data["Var5"].apply(lambda x:predict_class(x,summ,5)))
    accuracy=round(classifaccuracy(predict_data),2)
    errrate=round(classiferror(predict_data),2)
    print('Case 3: X=F2 Accuracy:',accuracy,' and Error:',errrate)
    
##########################################################################
#Case4: Training with first 100 data points using Z1,F2
#Calculate mean and standard deviation for first 100 data points of F2
#Use it to calculate probability of p(F1|Ci) but using F2 mean and std
#########################################################################    
    summZ1=summary_dataset(normdf[0:100])
    summdf2=summary_dataset(df2[0:100])
    predict_data=df1[100:]
    
    predict_data=predict_data.assign(PredictedValC1=predict_data["Var1"].apply(lambda x:predict_class_multiv(x,summZ1,summdf2,5)))
    predict_data=predict_data.assign(PredictedValC2=predict_data["Var2"].apply(lambda x:predict_class_multiv(x,summZ1,summdf2,5)))
    predict_data=predict_data.assign(PredictedValC3=predict_data["Var3"].apply(lambda x:predict_class_multiv(x,summZ1,summdf2,5)))
    predict_data=predict_data.assign(PredictedValC4=predict_data["Var4"].apply(lambda x:predict_class_multiv(x,summZ1,summdf2,5)))
    predict_data=predict_data.assign(PredictedValC5=predict_data["Var5"].apply(lambda x:predict_class_multiv(x,summZ1,summdf2,5)))
    accuracy=round(classifaccuracy(predict_data),2)
    errrate=round(classiferror(predict_data),2)
    print('Case 4: X=[Z1;F2] Accuracy:',accuracy,' and Error:',errrate)
    
    
main()