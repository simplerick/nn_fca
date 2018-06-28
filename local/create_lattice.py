import numpy as np
from fca import*
from data_loader import *



X, y, object_labels, attribute_labels = get_mammographic_masses()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_test, y_test = train_test_split(
    X, y_cl, tp=0.8)


fca = FCA(X_train)
fca.build_lattice()
fca.build_cover_relation()
fca.calculate_properties(y_cl)
fca.calculate_stability()
# fca.calculate_conf()
fca.save_lattice()
fca.save_properties()

