import numpy as np

name = input("Config file name: ")
configs = np.load("data/"+name+".npy")
print(len(configs))
dataset = input("Dataset: ")
n = int(input("Number of parts: "))

size = int(len(configs)/n)+1
for i in range(n):
    np.save("data/"+dataset+"_conf"+str(i), configs[ size*i : size*(i+1) ])