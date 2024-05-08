##Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;

from typing import Tuple, List
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

with open("AirQualityUCI.txt", "r") as file:
    training_data = [parse_line(line) for line in file.readlines() if len(line) > 4]

training_data = normalize(training_data)
train_data, test_data = train_test_split(training_data, test_size=0.25)

nn = NeuralNet(10, 5, 2)
nn.train(train_data, iters=5000, print_interval=500)

for i in nn.test_with_expected(test_data):
    difference = round(abs(i[1][0] - i[2][0]), 4)
    print(f"desired: {i[1]}, actual: {i[2]}, diff: {difference}")