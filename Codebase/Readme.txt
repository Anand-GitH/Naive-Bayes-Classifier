1. Converting matlab data file to excel data
T1 = table(F1(:,1),F1(:,2),F1(:,3),F1(:,4),F1(:,5))
filename = 'F1Data.xlsx';
writetable(T1,filename);

T2 = table(F2(:,1),F2(:,2),F2(:,3),F2(:,4),F2(:,5))
filename = 'F2Data.xlsx';
writetable(T2,filename);

2. Using this data to build classifier in python
mean=1/n*(sum(x1+----+xn))
std=sqrt(1/n-1*sum((xi-mean)^2))


