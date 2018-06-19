import numpy as np
import sys
from io import BytesIO
from tensorflow.python.lib.io import file_io
from . data_loader import *
from . neural_network import*


# import tensorflow as tf

data_set = sys.argv[1]
configs_file = sys.argv[2]
output_file =  configs_file


X, y, object_labels, attribute_labels = get_car_evaluation()[:4]
y_cl = one_hot(y, n_classes=4)
X_train, y_train, X_test, y_test = train_test_split(
    X, y_cl, tp=0.8)



# training params
tests = 15
num_epoch = 1000
optimizer="adam"
learning_rate = 0.001
batch_size = 100


f = BytesIO(file_io.read_file_to_string('gs://lattice_project/data/'+configs_file+'.npy',True))
configs = np.load(f)


for adj, res_connect, weights, conf in configs:
    try:
        results, times = model(adj,res_connect, weights, conf, X_train, y_train, X_test, y_test, prob = {}, optimizer=optimizer,learning_rate=learning_rate, batch_size=batch_size, tests=tests, num_epoch= num_epoch)
    except Exception:
        continue

    record = str(adj)+";"+"%.0f" % times.mean()+";"+";".join(list(map(lambda x: "%.2f" % x,results)))+";"+ "%.4f" % results.mean()
    o = file_io.FileIO('gs://lattice_project/data/'+output_file, 'a')
    o.write(record+'\n')      
                    






