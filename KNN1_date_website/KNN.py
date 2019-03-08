import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def file2matrix(filename):    #load data set
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


def y_classify(y):              # convert the string-type label to int
    for i in range(y.shape[0]):
        if y[i] == "didntLike":
            y[i] = 1
        elif y[i] == "smallDoses":
            y[i] = 2
        else:
            y[i] = 3
    return y

def plot_data(x,y):
    type1_x = []  # define three pairs of array to store the classification data
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []

    for i in range(len(y)):  # to classify the data
        if y[i] == '1':
            type1_x.append(x[i][0])
            type1_y.append(x[i][1])

        if y[i] == '2':
            type2_x.append(x[i][0])
            type2_y.append(x[i][1])

        if y[i] == '3':
            type3_x.append(x[i][0])
            type3_y.append(x[i][1])

    plt.scatter(type1_x, type1_y, s=20, c='r', label='didntLike')   # plot data
    plt.scatter(type2_x, type2_y, s=40, c='b', label='smallDoses')
    plt.scatter(type3_x, type3_y, s=60, c='k', label='largeDoses')
    plt.legend() # show plot
    plt.show()

def normalize(matrix):
    matrix_mean = sum(matrix)/len(matrix)
    min_value = matrix.min(0)
    max_value = matrix.max(0)
    ranges = max_value-min_value
    norm_matrix = (matrix-matrix_mean)/ranges
    return norm_matrix


def classify(x_train, y_train, x_test):
    y_label = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        dis = 100
        y_label[i] = 1
        for j in range(y_train.shape[0]):
            dis_temp = sum((x_test[i,:]-x_train[j,:])**2)
            if dis_temp < dis:
                dis = dis_temp
                y_label[i] = y_train[j]
    return y_label


x,y=file2matrix("datingTestSet.txt")
x = np.array(x)   # vectorized x and y
y = np.array(y)
y = y_classify(y)

fig = plt.figure()  #creat a empty plot
ax = fig.add_subplot(111)  #choose the location of the plot
plot_data(x,y)  # data visualization with the first two features

for i in range(x.shape[1]):  # normalize training data
    x[:,i] = normalize(x[:,i])

rand_arr = np.arange(x.shape[0])
np.random.shuffle(rand_arr)



y_result = classify(x[rand_arr[0:900],:],y[rand_arr[0:900]],x[rand_arr[900:1001],:])

y_result = y_result.astype(int)
yy = y[rand_arr[900:1001]].astype(int)


error_rate = len(np.flatnonzero(y_result-yy))/100


print(error_rate)

