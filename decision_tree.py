from util import entropy, information_gain, partition_classes
import numpy as np

class DecisionTree(object):
    def __init__(self):
        self.tree = {}

    def learn(self, X, y):
        self.tree = self.recursive_learn(X, y)

    def recursive_learn(self, X, y):
        # TODO: train decision tree and store it in self.tree

        if len(set(y)) == 1:
            aDict = {}
            aDict["value"] = y[0]
            return aDict
        if len(list(X)) == 0:
            aDict = {}
            aDict["value"] = np.argmax(y)
            return aDict

        attribute_index = 0
        ##attribute_threshold = None
        infoGain = 0
        sample_point = 0

        for index in range(0, len(X[0])):
            attribute = []
            for sample in range(len(X)):
                attribute.append(X[sample][index])

            maximum = max(attribute)

            for num in range(maximum):
                partitions = partition_classes(attribute, y, num)
                informationGain = information_gain(y, partitions)
                if infoGain < informationGain:
                    infoGain = informationGain
                    sample_point = num
                    attribute_index = index

        list1, list2, list3 = ([] for i in range(3))

        for record in X:
            list3.append(record[attribute_index])
            if record[attribute_index] <= sample_point:
                list1.append(record)
            else:
                list2.append(record)

        labels = partition_classes(list3, y, sample_point)
        aDict = {}
        aDict["current"] = (attribute_index, sample_point)
        aDict["left"] = self.recursive_learn(list1, labels[0])
        aDict["right"] = self.recursive_learn(list2, labels[1])
        return aDict

    def classify(self, record):
        # TODO: return predicted label for a single record using self.tree
        root = self.tree
        while "value" not in root.keys():
            attr = root["current"][0]
            sp = root["current"][1]
            if record[attr] <= sp:
                root = root["left"]
            else:
                root = root["right"]
        return root["value"]
















