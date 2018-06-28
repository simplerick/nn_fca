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


X, y, object_labels, attribute_labels = get_mammographic_masses()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_test, y_test = train_test_split(X, y_cl, tp=0.8)



# training params
tests = 20
num_epoch = 300
optimizer="adam"
learning_rate = 0.003
batch_size = 100


f = BytesIO(file_io.read_file_to_string('gs://lattice_project/data/'+configs_file+'.npy',True))
configs = np.load(f)


for adj, res_connect, weights, conf in configs:
    for init in (1,2,3,4,5,6):
        results, times = model(adj,{}, weights, X_train, y_train, X_test, y_test, conf, init, optimizer=optimizer,learning_rate=learning_rate, batch_size=batch_size, tests=tests, num_epoch= num_epoch)
        record = str(init) +";"+"%.0f" % times.mean()+";"+";".join(list(map(lambda x: "%.3f" % x,results)))+";"+ "%.4f" % results.mean()
        o = file_io.FileIO('gs://lattice_project/data/'+output_file, 'a')
        o.write(record+'\n')
    for init in (2,3,5,6):
        results, times = model(adj,res_connect, weights, X_train, y_train, X_test, y_test, conf, init, optimizer=optimizer,learning_rate=learning_rate, batch_size=batch_size, tests=tests, num_epoch= num_epoch)
        record = "resnet;"+str(init) +";"+"%.0f" % times.mean()+";"+";".join(list(map(lambda x: "%.3f" % x,results)))+";"+ "%.4f" % results.mean()
        o = file_io.FileIO('gs://lattice_project/data/'+output_file, 'a')
        o.write(record+'\n') 

         
                    






