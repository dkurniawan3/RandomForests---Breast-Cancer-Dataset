from scipy import stats
import numpy as np

def entropy(class_y):

    entropyOfList = 0
    unique, counts = np.unique(class_y, return_counts = True)
    aDict = dict(zip(unique, counts))
    for key in aDict:
        entropyOfList += -1*(np.log2(float(aDict[key])/(float(len(class_y))))*float(aDict[key])/(float(len(class_y))))
    return entropyOfList

def partition_classes(x, y, split_point):

    finalList = []
    smaller = []
    greater = []

    for element in zip(x,y):
        if element[0] <= split_point:
            smaller.append(element[1])
        else:
            greater.append(element[1])

    finalList.append(smaller)
    finalList.append(greater)
    return finalList

def information_gain(previous_y, current_y):

    entropyPrev = entropy(previous_y)
    entropyCurr = 0
    length = 0

    for element in current_y:
        length += len(element)

    for i in current_y:
        fraction = float(len(i))/float(length)
        entropyCurr += entropy(i) * fraction

    infoGain = entropyPrev - entropyCurr
    return infoGain







