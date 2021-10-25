SIZE_X = SIZE_Y = 3
BIAS = 1
EPOCHS = 20000
LEARN_RATE = 0.2

import random
import math


def intialize_weight(size):
    # x[0] is bias weight
    x = []
    x.append(random.uniform(-1, 1))
    for i in range(0,size,1):
        x.append(random.uniform(-1, 1)) 
    return x

def _get_result_perceptron(data,prediction, weights, bias, size_X, size_Y):
    result = bias * weights[0]
    x=1
    for i in range(size_X):
        for j in range(size_Y):
            result += data[i][j] * weights[x]
            x+=1
    return sign(result)

def _update_weights(data, weights, bias,result, expected, learn_rate, size_x, size_y):
    weights[0] = weights[0] + learn_rate * bias * (expected - result)
    w = 1 
    for x in range(size_x):
        for y in range(size_y):
            weights[w] = weights[w] + learn_rate * data[x][y] * (expected - result) 

def train(data_set, prediction_set, weights, bias, learn_rate, size_x, size_y, epochs, display=False):
    for epoch in range(epochs):
        epoch_sucess_rate = 0 
        if display:
            print('\n\n')
            print(('=' * 10) + f' EPOCH ({epoch}) ' + ('=' * 10))
            print('\n')
        for i in range(len(data_set)):
            if display:
                print(('*' * 10) + f' DATASET ITEM: ({i}) ' + ('*' * 10))
            result =_get_result_perceptron(data_set[i], prediction_set[i], weights, bias, size_x, size_y)
            _update_weights(data_set[i], weights, bias,result, prediction_set[i], learn_rate, size_x, size_y)
            if display:
                print(f'Expected:({prediction_set[i]}) | Result:({result}) ')
            if prediction_set[i] == result:
                epoch_sucess_rate += 1
        if display:
            print('\n')
            print(('-' * 10) + f' Weights ' + ('-' * 10))
            for w in range(len(weights)):
                print(('-' * 10) + f' Weight ({w}): {weights[w]}')
            print('\n')
            print(('-' * 6) + f' Prediction Success Rate ({(epoch_sucess_rate / len(data_set) * 100)}%) ' + ('-' * 6))
        if (epoch_sucess_rate / len(data_set)) == 1:
            break


def test(data, weights, bias, size_X, size_Y):
    result = bias * weights[0]
    x=1
    for i in range(size_X):
        for j in range(size_Y):
            result += data[i][j] * weights[x]
            x+=1
    return sign(result)

sign = lambda x: math.copysign(1, x)

data_set = [ 
    [[1,1,1],[0,1,0],[0,1,0]],
    [[1,0,1],[1,1,1],[1,0,1]]
    ]
    
data_set_result = [1,-1]

weights = intialize_weight(SIZE_X * SIZE_Y)

train(data_set,data_set_result, weights, BIAS, LEARN_RATE, SIZE_X, SIZE_Y, EPOCHS)

print(f'Test Result For T is: ({test([[1,1,1],[0,1,0],[0,1,0]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for H is: ({test([[1,0,1],[1,1,1],[1,0,1]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for T different is: ({test([[1,1,1],[0,1,1],[0,1,0]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for T different is: ({test([[1,1,1],[1,1,0],[0,1,0]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for H different is: ({test([[1,0,1],[1,0,1],[1,0,1]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for H different is: ({test([[1,0,1],[1,1,0],[1,0,1]], weights, BIAS, SIZE_X, SIZE_Y)})')
print(f'Test Result for H different is: ({test([[1,0,1],[0,1,1],[1,0,1]], weights, BIAS, SIZE_X, SIZE_Y)})')
