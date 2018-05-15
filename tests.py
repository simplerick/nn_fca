import numpy as np
from fca import*
from data_loader import *
from neural_network import*

# import tensorflow as tf


X, y, object_labels, attribute_labels = get_titanic()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(
    X, y_cl, tp=0.6, vp=0.2)


fca = FCA(X_train)
fca.load_lattice()
fca.calculate_properties(y_train)
fca.build_cover_relation()
fca.calculate_stability()
fca.save_partition()


# training params

tests = 5
num_epoch = 1000
optimizer="adam"
learning_rate = 0.13

# lower_supp = 0
# upper_supp = 0.3
# purity_value = 0.7
# accuracy_value = 0.3
# f_measure_value = 0.2

Stability_values = np.arange(0,0.9,0.1)
Accuracy_values = np.arange(0.2,0.5,0.1)
Purity_values = np.arange(0.5,0.9,0.1)
F_measure_values = np.arange(0,0.3,0.1)
Upper_supp = np.arange(0.3,0.7,0.1)
Lower_supp = np.arange(0,0.2,0.1)


for stability_value in Stability_values:
    print("\n\n Stability"+str(stability_value)+"\n\n")
    for upper_supp in Upper_supp:
        for lower_supp in Lower_supp:
            for purity_value in Purity_values:
                for accuracy_value in Accuracy_values:
                    for f_measure_value in F_measure_values:
                        fca.load_partition()
                        fca.load_lattice()
                        good_concepts = fca.select_concepts(lower_supp, upper_supp,  purity_value, accuracy_value, f_measure_value, stability_value)
                        fca.reduce_lattice(good_concepts)
                        adj = fca.build_cover_relation(reverse=True)

                        if len(good_concepts) > 25:
                            continue

                        config = str(lower_supp)+","+str(upper_supp)+","+str(purity_value)+","+str(accuracy_value)+","+str(f_measure_value)+","+str(stability_value)+","+str(len(good_concepts))+","+str(optimizer)+","+str(learning_rate)+","+str(tests)+","+str(num_epoch)

                        res_connect = dict((c,[]) for c in range(y_train.shape[1]))
                        for i,c in fca.partition_to_classes.items():
                            res_connect[c].append(i)

                        results, times = model(adj,res_connect, X_train, y_train, X_test, y_test, {}, optimizer,learning_rate, tests, num_epoch, config=config)
                        record = config+","+"%.0f" % times.mean()+","+",".join(list(map(lambda x: "%.2f" % x,results)))+","+ "%.4f" % results.mean()
                        f = open('results','a')
                        f.writelines(record+'\n')








# model(adj,res_connect, X_train, y_train, X_test, y_test, prob,learning_rate=0.05, tests=2, config="")
# model(adj,res_connect, X_train, y_train, X_test, y_test, prob,learning_rate=0.2, tests=2, config="")

# good_concepts, res_connect = fca.select_concepts(y_cl, lower_supp, upper_supp,  purity_value, accuracy_value, f_measure_value)
# fca.reduce_lattice(good_concepts)
# adj = fca.build_cover_relation(reverse=True)
# prob = fca.calculate_conf()

#   print("Number of concepts: " + str(len(fca.lattice)))
#   print(res_connect)
# model(learning_rate=0.05,lower_supp=0, upper_supp = 0.6,  purity_value = 0.87, accuracy_value = 0.4, f_measure_value=0.2,config="3_")
# model(learning_rate=0.05,lower_supp=0, upper_supp = 0.7,  purity_value = 0.85, accuracy_value = 0.4, f_measure_value=0.2,config="4_")