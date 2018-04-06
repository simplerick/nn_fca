import numpy as np
import random


def one_hot(y, n_classes=2):
    onehot = np.zeros(shape=(y.shape[0], n_classes), dtype='float32')

    onehot[np.arange(y.shape[0]), y] = 1.0
    return onehot


def train_test_split(X, y, tp=0.6, vp=None):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = int(n*tp)
    X_train = X[:train_size, :]
    y_train = y[:train_size]

    if vp is None:
        X_test = X[train_size:, :]
        y_test = y[train_size:]
        return X_train, y_train, X_test, y_test
    else:
        val_size = int(n*vp)
        X_val = X[train_size:(train_size+val_size), :]
        y_val = y[train_size:(train_size + val_size)]

        X_test = X[(train_size+val_size):, :]
        y_test = y[(train_size+val_size):]
        return X_train, y_train, X_val, y_val, X_test, y_test


def get_titanic(file_name='data_sets//titanic.txt'):
    data = [row.split(';') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    attributes = np.array(data[0][1:-1])
    objects = np.array([row[0] for row in data[1:]])
    y_label = data[0][-1]
    X = np.array([[int(s) for s in row[1:-1]] for row in data[1:]])
    y = np.array([int(row[-1]) for row in data[1:]])

    return X, y, objects, attributes, y_label


def get_zoo(file_name='data_sets//zoo.txt'):
    data = [row.split(',') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    attributes = np.array([
        'hair',
        'feathers',
        'eggs',
        'milk',
        'airborne',
        'aquatic',
        'predator',
        'toothed',
        'backbone',
        'breathes',
        'venomous',
        'fins',
        'legs:0',
        'legs:2',
        'legs:4',
        'tail',
        'domestic',
        'catsize'
    ])
    objects = np.array([row[0] for row in data[1:]])
    y_label = 'type'
    X = []
    y = []
    for row in data[1:]:
        try:
            X.append([
                1 if int(row[1]) > 0 else 0,
                1 if int(row[2]) > 0 else 0,
                1 if int(row[3]) > 0 else 0,
                1 if int(row[4]) > 0 else 0,
                1 if int(row[5]) > 0 else 0,
                1 if int(row[6]) > 0 else 0,
                1 if int(row[7]) > 0 else 0,
                1 if int(row[8]) > 0 else 0,
                1 if int(row[9]) > 0 else 0,
                1 if int(row[10]) > 0 else 0,
                1 if int(row[11]) > 0 else 0,
                1 if int(row[12]) > 0 else 0,
                1 if int(row[13]) == 0 else 0,
                1 if int(row[13]) == 2 else 0,
                1 if int(row[13]) >= 4 else 0,
                1 if int(row[14]) > 0 else 0,
                1 if int(row[15]) > 0 else 0,
                1 if int(row[16]) > 0 else 0
            ])
            y.append(int(row[-1])-1)
        except:
            pass
    X = np.array(X)
    y = np.array(y)

    return X, y, objects, attributes, y_label


def get_breast_cancer(file_name='data_sets//breast-cancer-wisconsin.txt'):
    data = [row.split(',') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    attributes = np.array([
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses'
    ])
    objects = np.array([row[0] for row in data[1:]])
    y_label = 'Class'
    X = []
    y = []
    for row in data[1:]:
        try:
            X.append([
                1 if int(row[1]) > 5 else 0,
                1 if int(row[2]) > 5 else 0,
                1 if int(row[3]) > 5 else 0,
                1 if int(row[4]) > 5 else 0,
                1 if int(row[5]) > 5 else 0,
                1 if int(row[6]) > 5 else 0,
                1 if int(row[7]) > 5 else 0,
                1 if int(row[8]) > 5 else 0,
                1 if int(row[9]) > 5 else 0
            ])
            y.append(0 if int(row[-1]) == 2 else 1)
        except:
            pass
    X = np.array(X)
    y = np.array(y)

    return X, y, objects, attributes, y_label


def get_seismic_bumps(file_name='data_sets//seismic-bumps.txt'):
    data = [row.split(',') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    objects = np.array([str(i) for i in range(len(data))])
    y_label = data[0][-1]
    X = []
    y = []
    for row in data:
        try:
            X.append([
                1 if row[0] in {'c', 'd'} else 0,
                1 if row[0] in {'a', 'b'} else 0,
                1 if row[1] in {'c', 'd'} else 0,
                1 if row[1] in {'a', 'b'} else 0,
                1 if row[2] == 'W' else 0,
                1 if int(row[3]) > 100000 else 0,
                1 if int(row[4]) > 100 else 0,
                1 if int(row[5]) > 0 else 0,
                1 if int(row[6]) > 0 else 0
            ])
            y.append(int(row[-1]))
        except:
            pass
    X = np.array(X)
    attributes = np.array([str(i) for i in range(X.shape[1])])
    y = np.array(y)

    return X, y, objects, attributes, y_label


def get_car_evaluation(file_name='data_sets//car.txt'):
    data = [row.split(',') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    objects = [str(i) for i in range(len(data)-1)]
    y_label = data[0][-1]
    X = []
    y = []
    for row in data[1:]:
        try:
            X.append([
                1 if int(row[0]) > 0 else 0,
                1 if int(row[0]) > 1 else 0,
                1 if int(row[0]) > 2 else 0,
                1 if int(row[1]) > 0 else 0,
                1 if int(row[1]) > 1 else 0,
                1 if int(row[1]) > 2 else 0,
                1 if int(row[2]) > 2 else 0,
                1 if int(row[2]) > 3 else 0,
                1 if int(row[2]) > 4 else 0,
                1 if int(row[3]) > 2 else 0,
                1 if int(row[3]) > 4 else 0,
                1 if int(row[4]) > 0 else 0,
                1 if int(row[4]) > 1 else 0,
                1 if int(row[5]) > 0 else 0,
                1 if int(row[5]) > 1 else 0
            ])
            y.append(int(row[-1]))
        except:
            pass
    X = np.array(X)
    attributes = [str(i) for i in range(X.shape[1])]
    y = np.array(y)

    return X, y, objects, attributes, y_label


def get_mammographic_masses(file_name='data_sets//mammographic_masses.txt'):
    data = [row.split(',') for row in open(file_name, 'r').read().strip('\n').split('\n')]
    attributes = np.array([
        'BI-RADS',
        'Age:20',
        'Age:30',
        'Age:45',
        'Shape:1',
        'Shape:2',
        'Shape:3',
        'Shape:4',
        'Margin:1',
        'Margin:2',
        'Margin:3',
        'Margin:4',
        'Margin:5',
        'Density:1',
        'Density:2',
        'Density:3'
    ])
    objects = [str(i) for i in range(len(data)-1)]
    y_label = data[0][-1]
    X = []
    y = []
    for row in data[1:]:
        try:
            X.append([
                1 if int(row[0]) > 3 else 0,
                1 if int(row[1]) < 20 else 0,
                1 if int(row[1]) < 30 else 0,
                1 if int(row[1]) < 45 else 0,
                1 if int(row[2]) == 1 else 0,
                1 if int(row[2]) == 2 else 0,
                1 if int(row[2]) == 3 else 0,
                1 if int(row[2]) == 4 else 0,
                1 if int(row[3]) == 1 else 0,
                1 if int(row[3]) == 2 else 0,
                1 if int(row[3]) == 3 else 0,
                1 if int(row[3]) == 4 else 0,
                1 if int(row[3]) == 5 else 0,
                1 if int(row[4]) <= 1 else 0,
                1 if int(row[4]) <= 2 else 0,
                1 if int(row[4]) <= 3 else 0
            ])
            y.append(int(row[-1]))
        except:
            pass
    X = np.array(X)
    y = np.array(y)

    return X, y, objects, attributes, y_label


def get_random(n=1000, m=15, frequency=0.5):
    index_ones = random.sample(range(n*m), int(frequency*n*m))
    X = np.zeros((n, m))
    for index in index_ones:
        X[index//m, index%m] = 1
    y = np.rint(np.random.uniform(size=n)).astype(dtype='int')

    objects = [str(i) for i in range(n)]
    attributes = [str(i) for i in range(m)]

    y_label = 'class'

    return X, y, objects, attributes, y_label
