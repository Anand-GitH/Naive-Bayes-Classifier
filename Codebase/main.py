######################################################################
#Main - which includes 
#Reading data from mat data file
#Calculating mean,std and then normalizing data
#Predict class and accuracy of each case
######################################################################

import pandas as pd
from genericcalc import summary_dataset
from genericcalc import predict_class
from genericcalc import classifaccuracy
from genericcalc import classiferror
from genericcalc import znormal_row
from genericcalc import predict_class_multiv
from readwritefiles import read_matdata
from plotresults import plot_features
from plotresults import plot_corwrgpreds
from plotresults import plot_accerror

def main():
    acc_list=[]
    err_list=[]
    matdata=read_matdata("data_EAS595.mat")
    
    df1=pd.DataFrame(matdata['F1'])
    df2=pd.DataFrame(matdata['F2'])
    
    df1=df1.rename(columns={0: 'C1',1:'C2',2:'C3',3:'C4',4:'C5'})
    df2=df2.rename(columns={0: 'C1',1:'C2',2:'C3',3:'C4',4:'C5'})
    normdf=znormal_row(df1)
    
    plot_features(df1,df2,"Feature Distribution","F1-1st Feature","F2-2nd Feature")
    plot_features(normdf,df2,"Feature Distribution-(F1 Normalized- Z1)","Z1-Normalized 1st Feature","F2-2nd Feature")
##########################################################################
#Case1: Training with first 100 data points 
#Calculate mean and standard deviation for first 100 data points
#Use it to calculate probability of p(F1|Ci)
#########################################################################
    summ=summary_dataset(df1[0:100])
    test_data=df1[100:]
    test_data=test_data.assign(PredictedValC1=test_data["C1"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC2=test_data["C2"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC3=test_data["C3"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC4=test_data["C4"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC5=test_data["C5"].apply(lambda x:predict_class(x,summ,5)))
    
    accuracy_data=classifaccuracy(test_data)
    error_data=classiferror(test_data)
    accuracy=round(accuracy_data[0],4)*100
    errrate=round(error_data[0],4)*100
    acc_list.append(accuracy)
    err_list.append(errrate)
    print('Case 1: X=F1 Accuracy:',accuracy,' and Error:',errrate)
    plot_corwrgpreds(accuracy_data[1],error_data[1]," for case X=F1")
##########################################################################
#Case2: Training with first 100 data points on normalized F1
#Calculate mean and standard deviation for first 100 data points which will 
#be zero and sd will be 1 as data is normalized
#Use it to calculate probability of p(Z1|Ci)
#########################################################################
    normdf=znormal_row(df1)
    summ=summary_dataset(normdf[0:100])
    test_data=normdf[100:]
    test_data=test_data.assign(PredictedValC1=test_data["C1"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC2=test_data["C2"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC3=test_data["C3"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC4=test_data["C4"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC5=test_data["C5"].apply(lambda x:predict_class(x,summ,5)))
    
    accuracy_data=classifaccuracy(test_data)
    error_data=classiferror(test_data)
    accuracy=round(accuracy_data[0],4)*100
    errrate=round(error_data[0],4)*100
    acc_list.append(accuracy)
    err_list.append(errrate)
    print('Case 2: X=Z1 Accuracy:',accuracy,' and Error:',errrate)
    plot_corwrgpreds(accuracy_data[1],error_data[1]," for case X=Z1")
##########################################################################
#Case3: Training with first 100 data points using F2
#Calculate mean and standard deviation for first 100 data points of F2
#Use it to calculate probability of p(F2|Ci) but using F2 mean and std
#########################################################################    
    summ=summary_dataset(df2[0:100])
    test_data=df2[100:]
    test_data=test_data.assign(PredictedValC1=test_data["C1"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC2=test_data["C2"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC3=test_data["C3"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC4=test_data["C4"].apply(lambda x:predict_class(x,summ,5)))
    test_data=test_data.assign(PredictedValC5=test_data["C5"].apply(lambda x:predict_class(x,summ,5)))

    accuracy_data=classifaccuracy(test_data)
    error_data=classiferror(test_data)
    accuracy=round(accuracy_data[0],4)*100
    errrate=round(error_data[0],4)*100
    acc_list.append(accuracy)
    err_list.append(errrate)
    print('Case 3: X=F2 Accuracy:',accuracy,' and Error:',errrate)
    plot_corwrgpreds(accuracy_data[1],error_data[1]," for case X=F2")
##############################################################################
#Case4: Training with first 100 data points using Z1,F2
#Calculate mean and standard deviation for first 100 data points of F2 and Z1
#Use it to calculate probability of p(Z1|Ci)*P(F2|Ci) assuming independence
#############################################################################   
    normdf=znormal_row(df1)
    summZ1=summary_dataset(normdf[0:100])
    summdf2=summary_dataset(df2[0:100])
    
    ztest_data=normdf[100:]
    f2test_data=df2[100:]
        
    PredictValC1=[]
    PredictValC2=[]
    PredictValC3=[]
    PredictValC4=[]
    PredictValC5=[]
    
    zdatac1=ztest_data["C1"]
    f2datac1=f2test_data["C1"]
    zdatac2=ztest_data["C2"]
    f2datac2=f2test_data["C2"]
    zdatac3=ztest_data["C3"]
    f2datac3=f2test_data["C3"]
    zdatac4=ztest_data["C4"]
    f2datac4=f2test_data["C4"]
    zdatac5=ztest_data["C5"]
    f2datac5=f2test_data["C5"]
            
    for i in (ztest_data.index):
        PredictValC1.append(predict_class_multiv(zdatac1[i],f2datac1[i],summZ1,summdf2,5))
        PredictValC2.append(predict_class_multiv(zdatac2[i],f2datac2[i],summZ1,summdf2,5))
        PredictValC3.append(predict_class_multiv(zdatac3[i],f2datac3[i],summZ1,summdf2,5))        
        PredictValC4.append(predict_class_multiv(zdatac4[i],f2datac4[i],summZ1,summdf2,5))        
        PredictValC5.append(predict_class_multiv(zdatac5[i],f2datac5[i],summZ1,summdf2,5))
        
    ztest_data=ztest_data.assign(PredictedValC1=PredictValC1,PredictedValC2=PredictValC2,PredictedValC3=PredictValC3,PredictedValC4=PredictValC4,PredictedValC5=PredictValC5)
    
    accuracy_data=classifaccuracy(ztest_data)
    error_data=classiferror(ztest_data)
    accuracy=round(accuracy_data[0],4)*100
    errrate=round(error_data[0],4)*100
    acc_list.append(accuracy)
    err_list.append(errrate)
    print('Case 4: X=[Z1;F2] Accuracy:',accuracy,' and Error:',errrate)
    plot_corwrgpreds(accuracy_data[1],error_data[1]," for case X=[Z1;F2]")
    plot_accerror(acc_list,err_list)
main()