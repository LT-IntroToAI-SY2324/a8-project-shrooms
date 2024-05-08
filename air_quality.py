##Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;

from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

def parse_line(line: str) -> Tuple[List[float], List[float]]:
 
    tokens = line.split(";")
    inpt = [float(x) for x in tokens[2:12]]
    output = [float(x) for x in tokens[13:15]]
    return (inpt, output)

def normalize(data: List[Tuple[List[float], List[float]]]):

    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

with open("AirQualityUCI.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

td = normalize(training_data)
print(td[0])
print(len(td))

nn = NeuralNet(11, 0, 2)

train_data, test_data = train_test_split(td)
# print(len(train_data))
# print(len(test_data))
# print(test_data)

nn.train(train_data, iters=1000, print_interval=100)