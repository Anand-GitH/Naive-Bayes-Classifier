# Naive-Bayes-Classifier
Naive Bayes classifier for continuous data using python

In an experiment involving 1000 participants, we recorded two different measurement (𝐹1 and 𝐹2) while participants performed 5 different tasks (𝐶1, 𝐶2, ⋯ 𝐶5). The two measurements are independent and for
each class they can be considered to have a normal distribution as follow:
𝑃(𝐹1|𝐶𝑖) = 𝑁(𝑚1𝑖, 𝜎1𝑖2) and 𝑃(𝐹2|𝐶𝑖) = 𝑁(𝑚2𝑖, 𝜎2𝑖2) for 𝑖 = 1,2, ⋯ 5where, 𝑚1𝑖, 𝜎1𝑖2 are the mean and variance of 𝐹1 for the ith class. 
Similarly, 𝑚2𝑖, 𝜎2𝑖2 are the mean and variance of 𝐹2 for the ith class. The goal of this project is to construct a classifier such that for any given values of 𝐹1 and 𝐹2, it can predict the performed task (𝐶1, 𝐶2, ⋯ 𝐶5). Let’s assume that the classifier calculate the probability of each class
given the measurement data, and output the most probable class as the predicted class.
𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝐶𝑙𝑎𝑠𝑠 = argmax[𝑃(𝐶𝑖|𝑋 )] , 𝑖 = 1,2, ⋯ 5
The file ‘data.m’ contains measurements F1 and F2 that are both matrices with the size of 1000x5. Each
column contains the information of one of the subjects and each row corresponds to one of the tasks (1st
row: 1st task, 2nd row: 2nd task, etc.)
