# Naive Bayes classifier for continuous data
Naive Bayes classifier for continuous data using python

In an experiment involving 1000 participants, we recorded two different measurement (𝐹1 and 𝐹2) while
participants performed 5 different tasks (𝐶1, 𝐶2, ⋯ 𝐶5). The two measurements are independent and for
each class they can be considered to have a normal distribution as follow:
𝑃(𝐹1|𝐶𝑖) = 𝑁(𝑚1𝑖, 𝜎1𝑖2) and 𝑃(𝐹2|𝐶𝑖) = 𝑁(𝑚2𝑖, 𝜎2𝑖2) for 𝑖 = 1,2, ⋯ 5 where, 𝑚1𝑖, 𝜎1𝑖2 are the mean and variance of 𝐹1 for the ith class. Similarly, 𝑚2𝑖, 𝜎2𝑖2 are the mean and variance of 𝐹2 for the ith class.
The goal of this project is to construct a classifier such that for any given values of 𝐹1 and 𝐹2, it can predict
the performed task (𝐶1, 𝐶2, ⋯ 𝐶5). Let’s assume that the classifier calculate the probability of each class
given the measurement data, and output the most probable class as the predicted class.
𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝐶𝑙𝑎𝑠𝑠 = argmax[𝑃(𝐶𝑖|𝑋 )] , 𝑖 = 1,2, ⋯ 5
The file ‘data.m’ contains measurements F1 and F2 that are both matrices with the size of 1000x5. Each
column contains the information of one of the subjects and each row corresponds to one of the tasks (1st
row: 1st task, 2nd row: 2nd task, etc.)
To find the best classifier, perform the following tasks:
 Step 1.Training: Use the data of the first 100 subjects to estimate 𝑚1𝑖, 𝜎1𝑖2 and 𝑚2𝑖, 𝜎2𝑖2
 Step 2.1.Testing: Assume that 𝑋 = 𝐹1. Using the Bayes' theorem, calculate the probability of each class for data of the remaining subjects (columns 101-1000 of 𝐹1) and consequently predict the class for each data point. Note that each subject performed 5 different tasks so you need to predict the class of 4500 data points.
 Step 2.2.Calculating the accuracy of the classifier: You need to check the percentage of the data whose class are correctly predicted. The true class is the row number of the data. So if you classify F1(3,131) as class 3, it is correctly classified otherwise you have wrongly predict the class.
Classification accuracy = correct predictions / total predictions (which is 4500 in this case)
Error rate = incorrect predictions / total predictions
 Step 3. Standard Normal (Z-Score): Assume 𝐹1 to be a subjective measure. In this case the mean value and the range of data reported by one subject will not be consistent with another subject. In other to remove the effect of individual differences, you have to normalize the data of each subject using the standard normal formulation (removing the mean and dividing by standard deviation). Calculate 𝑍1 (the standard normal of 𝐹1) and plot the distribution of the data using 𝑍1 and 𝐹2, and compare it to the distribution in 𝐹1 and 𝐹2 
 Step 4.Repeat 2.1 and 2.2 for the following cases:
o Case 2: 𝑋 = 𝑍1 (Note for this case you need to repeat the training step as well)
o Case 3: 𝑋 = 𝐹2
o Case 4: 𝑋 = [𝑍1 𝐹2]. Note that this is a multivariate normal distribution and you need to
use the independence assumption.
  Step 5. Compare the classification rate of the four cases
