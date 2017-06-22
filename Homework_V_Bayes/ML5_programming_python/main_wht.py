import pickle
import math

import sys, time

def readData():
    X = pickle.load(open('train_data.pkl', 'rb')).todense().tolist()
    y = pickle.load(open('train_targets.pkl', 'rb')).tolist()
    Xt = pickle.load(open('test_data.pkl', 'rb')).todense().tolist()
    return X, y, Xt


def separateByClass(X, y):
    separated = {}
    for i in range(len(X)):
        if (y[i] not in separated):
            separated[y[i]] = []
        separated[y[i]].append(X[i])
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    dataset_transpose = list(zip(*dataset))
    discrete_data = dataset_transpose[:2500]
    continuous_data = dataset_transpose[2500:]
    summaries = [[(sum(attribute) + 1) / float(len(attribute) + 2)] for attribute in discrete_data]
    summaries += [[mean(attribute), stdev(attribute)] for attribute in continuous_data]
    return summaries


def summarizeByClass(dataset, y):
    separated = separateByClass(dataset, y)
    summaries = {}
    priori = {}
    for classValue, instances in iter(separated.items()):
        summaries[classValue] = summarize(instances)
        priori[classValue] = len(instances) / float(len(y))
    return summaries, priori


def calculateProbability(x, type, p_1=0, mean=0, stdev=1):
    if type == 'cont':
        if stdev < 1e-9:
            return float(x == mean)
        else:
            exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
            return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    else:
        if x == 0:
            return 1 - p_1
        else:
            return p_1


def robust_log(x):
    if x == 0:
        return -float('Inf')
    else:
        return math.log(x)


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in iter(summaries.items()):
        probabilities[classValue] = 0
        for i in range(len(classSummaries)):
            if i < 2500:
                (p1, ) = classSummaries[i]
                x = inputVector[i]
                # print(robust_log(calculateProbability(x, 'disc', p_1=p1)))
                probabilities[classValue] += robust_log(calculateProbability(x, 'disc', p_1=p1))
            # else:
            #     sys_exit()
            #     mean, stdev = classSummaries[i]
            #     x = inputVector[i]
            #     probabilities[classValue] += robust_log(calculateProbability(x, 'cont', mean=mean, stdev=stdev))
    return probabilities


def predict(summaries, priori, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    for classValue, classProb in iter(probabilities.items()):
        probabilities[classValue] += robust_log(priori[classValue])
    # print(probabilities)
    bestLabel, bestProb = None, -float('Inf')
    for classValue, probability in iter(probabilities.items()):
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, priori, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, priori, testSet[i])
        predictions.append(result)
    return predictions

start_time = time.time()

X, y, Xt = readData()
summary, priori = summarizeByClass(X, y)
pred = getPredictions(summary, priori, Xt)
print(len(Xt), len(pred))
f = open('test_predictions.csv', 'w')
for pred_val in pred:
    f.write(str(pred_val) + '\n')
print('Cost time = ', time.time() - start_time)

