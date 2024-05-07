from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

def parse_line(line: str) -> Tuple[List[float], List[float]]:
 
    tokens = line.split(",")
    out = int(tokens[0])
    output = [0 if out == 1 else 0.5 if out == 2 else 1]

    inpt = [float(x) for x in tokens[1:]]
    return (inpt, output)
