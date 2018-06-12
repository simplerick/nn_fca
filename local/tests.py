import numpy as np
from data_loader import *
from neural_network import*
from fca import*




X, y, object_labels, attribute_labels = get_titanic()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_test, y_test = train_test_split(
    X, y_cl, tp=0.8)



configs = np.load("data/configs.npy")

# training params

tests = 5
num_epoch = 1000
optimizer="adam"
learning_rate = 0.001
batch_size = 100



for adj, res_connect, weights in configs[6:8]:
    
    results, times = model(adj,res_connect, weights, X_train, y_train, X_test, y_test, prob = {}, optimizer=optimizer,learning_rate=learning_rate, batch_size=batch_size, tests=tests, num_epoch= num_epoch)
    record = str(adj)+";"+"%.0f" % times.mean()+";"+";".join(list(map(lambda x: "%.2f" % x,results)))+";"+ "%.4f" % results.mean()
    f = open('data/results', 'a')
    f.write(record+'\n')      





# for upper_supp in Upper_supp:
#     print("\n\n Upper_supp"+str(upper_supp)+"\n\n")
#     for stability_value in Stability_values:
#         for purity_value in Purity_values:
#             fca.load_partition()
#             fca.load_lattice()
#             good_concepts = fca.select_concepts(lower_supp, upper_supp,  purity_value, accuracy_value, f_measure_value, stability_value)
#             fca.reduce_lattice(good_concepts)
#             adj = fca.build_cover_relation(reverse=True)

#             if len(good_concepts) > 35 or len(good_concepts) < 3:
#                 continue

#             if stability_value == Stability_values[0]:
#                 if fca.adj == prev_net2:
#                     break
#                 prev_net2 = fca.adj

#             if fca.adj == prev_net:
#                 continue
#             prev_net = fca.adj

#             print(len(good_concepts))

#             config = "%.2f" % upper_supp +","+ "%.2f" % stability_value +","+ "%.2f" % purity_value+","+str(lower_supp)+","+str(len(good_concepts))+","+str(optimizer)+","+str(learning_rate)+","+str(tests)+","+str(num_epoch)

#             res_connect = dict((c,[]) for c in range(y_train.shape[1]))
#             for i,c in fca.partition_to_classes.items():
#                 res_connect[c].append(i)

#             print(res_connect)

#             # try:
#             #     results, times = model(adj,res_connect, X_train, y_train, X_test, y_test, {}, optimizer,learning_rate, tests, num_epoch, config=config)
#             # except Exception:
#             #     continue
#             # record = config+","+"%.0f" % times.mean()+","+",".join(list(map(lambda x: "%.2f" % x,results)))+","+ "%.4f" % results.mean()
#             record = config
#             f = open('results','a')
#             f.writelines(record+'\n')
                        





