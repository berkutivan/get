import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from scipy import stats
import seaborn as sns
from math import sqrt
from math import pi
from math import exp
iris_dataset = load_iris()

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


#первый шаг - разделяем по классам
# data - iris_dataset['data'])
# target - iris_dataset['target']
def separate_by_class(data, target):
    separated = dict()
    for i in range(len(target)):
        vector = data[i]
        class_value = target[i]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

#обобщение набора данных и выведение статистики

def summarize_dataset(separated):
    class_statistics = dict()

    for object in separated:

        class_statistics[object] = list()
        params = np.array(separated[object])

        for i in range(params.shape[1]):
            X = params[:, i]
            Y = np.array([np.mean(X), np.std(X), len(X)])
            class_statistics[object].append(Y)

    return class_statistics




# вероятности
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities





#обучалка

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#separated = separate_by_class(iris_dataset['data'], iris_dataset['target'])
#print(iris_dataset['data'][0])
#print(calculate_class_probabilities(summarize_dataset(separated), iris_dataset['data'][0]))

#выборка
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0, train_size = 0.2)

#далее обучаем

separated = separate_by_class(X_train, Y_train)

predict = []

for i in X_test:
    a = calculate_class_probabilities(summarize_dataset(separated), i)
    maxx = 0
    answer = -1
    for j in a:
        if a[j] > maxx:
            maxx = a[j]
            answer = j
    predict.append(answer)

print(accuracy_metric(Y_test, predict))