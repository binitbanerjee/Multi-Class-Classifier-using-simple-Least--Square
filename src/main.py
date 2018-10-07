dataPath = '../data/iris.csv'
import pandas as pd
import numpy as np
import numpy.linalg as ln
import math

data = pd.read_csv(dataPath)
l = .004

def splitDataSamples(n):
    t1=[]
    t2=[]
    t3=[]
    for i in data.values:
        if i[4]=='setosa':
            t1.append(i)
        elif i[4]=='versicolor':
            t2.append(i)
        else:
            t3.append(i)
    t1L,t2L,t3L = len(t1),len(t2),len(t3)
    it1,it2,it3 = math.floor((t1L*n)/100),math.floor((t2L * n) / 100),math.floor((t3L * n) / 100)
    train = []
    test =[]
    samples =[]
    train.append(t1[0:it1])
    train.append(t2[0:it2])
    train.append(t3[0:it3])
    test.append(t1[it1+1:])
    test.append(t2[it2+1:])
    test.append(t3[it3+1:])
    samples.append(train)
    samples.append(test)
    return samples

def calculateCorrMatrix(dataSet):
    correlationMatrix = np.zeros((5,5))
    for item in dataSet:

        for j in item :
            x = []
            for i in j:
                try:
                    x.append([float(i)])
                except:
                    x.append([1.0])
            x = np.array(x)
            xt = np.transpose(x)
            prod = np.matmul(x,xt)
            correlationMatrix = correlationMatrix+prod
    return correlationMatrix

def calculateYX(dataSet):
    y=[[0,0,0]]
    sum = np.zeros((5,3))
    for item in dataSet:
        for j in item:
            x = []
            for i in j:
                try:
                    x.append([float(i)])
                except:
                    if(i=='setosa'):
                        y[0][0] = 1.
                    elif(i=='versicolor'):
                        y[0][1] = 1.
                    else:
                        y[0][2] = 1.
                    x.append([1.0])
            temp = np.copy(y)
            y = [[0,0,0]]
            sum = sum + np.matmul(x,temp)
    return sum


def calculateWeightMatrix(dataSet):
    cMatrix = calculateCorrMatrix(dataSet)
    interim = cMatrix + l
    t = ln.inv(np.array(interim))
    y = calculateYX(dataSet)
    tempWeight=np.matmul(t,y)
    return tempWeight

def normalizeClassifiedData(predictedData):
    maxValue = max(predictedData)
    result =[]
    for i in predictedData:
        if(i==maxValue):
            result.append(1)
        else:
            result.append(0)
    return result

def classifyRecord(record,weightMatrix):
    item = record[0:5]
    x = []
    for i in item:
        try:
            x.append([float(i)])
        except:
            x.append([1.0])
    prediction = np.matmul(np.transpose(weightMatrix),x)
    result = normalizeClassifiedData(prediction)
    return result

def testClassification(testData,weightMatrix):
    result = []
    predicted =""
    countOfCorrectPrediction = 0
    setosa = [0,0,0]
    versicolor = [0,0,0]
    verginica = [0,0,0]
    for item in testData:
        for i in item:
            prediction = classifyRecord(i,weightMatrix)
            actualName = i[4]
            if(prediction[0]==1):
                predicted = "setosa"
                temp = setosa
                index = 0
            elif (prediction[1] == 1):
                predicted = "versicolor"
                temp = versicolor
                index = 1
            elif (prediction[2] == 1):
                predicted = "virginica"
                temp = verginica
                index = 2
            if(predicted==actualName):
                countOfCorrectPrediction = countOfCorrectPrediction+1
                temp[index] = temp[index]+1
            else:
                if (predicted=='setosa'):
                    if(actualName =="versicolor"):
                        versicolor[index] = versicolor[index]+1
                    else:
                        verginica[index] = verginica[index] +1
                elif (predicted == "versicolor"):
                    if (actualName == "setosa"):
                        setosa[index] = setosa[index] + 1
                    else:
                        verginica[index] = verginica[index] + 1
                elif (predicted == "virginica"):
                    if (actualName == "setosa"):
                        setosa[index] = setosa[index] + 1
                    else:
                        versicolor[index] = versicolor[index] + 1
            predicted = ""
    l = len(testData[0])+len(testData[1])+len(testData[2])
    print("Total set : ",l)
    print("Total Correct prediction : ", countOfCorrectPrediction)
    print("Percentage accuracy : ",((countOfCorrectPrediction/l)*100))
    return setosa,versicolor,verginica

def calculateAndPrintConfusionMatrix(sampleSize):
    print("using ", sampleSize, "% for training")
    sampleData = splitDataSamples(sampleSize)
    meanWeightMatrix = calculateWeightMatrix(sampleData[0])
    setosa1,versicolor1,verginica1 = testClassification(sampleData[1],meanWeightMatrix)
    confusionMatrix = []
    confusionMatrix.append([setosa1,versicolor1,verginica1])
    print(confusionMatrix)

calculateAndPrintConfusionMatrix(12)
calculateAndPrintConfusionMatrix(50)
calculateAndPrintConfusionMatrix(30)




