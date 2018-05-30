import numpy as np
from fca import*
from data_loader import *



X, y, object_labels, attribute_labels = get_titanic()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(
    X, y_cl, tp=0.6, vp=0.2)




fca = FCA(X_train)
fca.build_lattice()
fca.build_cover_relation()
fca.calculate_properties(y_cl)
fca.calculate_stability()
fca.calculate_conf()
fca.save_lattice()
fca.save_properties()