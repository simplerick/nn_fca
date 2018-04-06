import numpy as np
import tensorflow as tf
from FCA import*
from data_loader import *
import time


X, y, object_labels, attribute_labels = get_titanic(
    file_name='data_sets//titanic.txt')[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(
    X, y_cl, tp=0.6, vp=0.2)


dim_in = X_train.shape[1]
dim_out = y_train.shape[1]


fca = FCA(X_train)
lattice = fca.build_lattice(level=2)
adj = fca.build_cover_relation()

print("Number of concepts: " + str(len(lattice)))


first_level = []
last_level = {}
weight = {}
bias = {}
# n = np.array([tf.Variable(0.) for _ in range(len(adj))])
n = np.array([tf.Variable(0.) for _ in range(len(adj))])

x = tf.placeholder(tf.float32, [None, dim_in])
y_ = tf.placeholder(tf.float32, [None, dim_out])


# initialization
print('Initialization')

last_level = set(adj.keys())

for i in adj:
    if len(adj[i]) == 0:
        first_level.append(i)
    for j in adj[i]:
        last_level.discard(j)
        weight[str(i) + ',' + str(j)
               ] = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
    bias[str(i)] = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))

last_level = list(last_level)

# weight['in'] = tf.Variable(tf.truncated_normal([dim_in,len(first_level)], stddev = 0.1))
# bias['in'] = tf.Variable(tf.truncated_normal([len(first_level)], stddev = 0.1))

# weight['out'] = tf.Variable(tf.truncated_normal([len(last_level), dim_out], stddev = 0.1))
# bias['out'] = tf.Variable(tf.truncated_normal([dim_out], stddev = 0.1))

for i in first_level:
    weight[str(i)] = tf.Variable(tf.truncated_normal([dim_in, 1], stddev=0.1))

for i in last_level:
    weight[str(i)] = tf.Variable(tf.truncated_normal([1, dim_out], stddev=0.1))

bias['out'] = tf.Variable(tf.truncated_normal([2], stddev=0.1))


# network connecting
print('Network connecting')

for i in first_level:
    n[i] = tf.nn.relu(tf.matmul(x, weight[str(i)]) + bias[str(i)])

def process(v):
    activation = tf.Variable(0.)
    if len(adj[v]) == 0:
        return
    for w in adj[v]:
        process(w)
        activation += weight[str(v) + ',' + str(w)] * n[w]
    n[v] = tf.nn.relu(activation + bias[str(v)])

for v in last_level:
    process(v)

y0 = bias['out']
for i in last_level:
    y0 += n[i] * weight[str(i)]

y = tf.nn.softmax(y0)


# training
print('Training')

tic = time.time()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train_step, feed_dict={x: X_train, y_: y_train})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

toc = time.time()

print("Accuracy: " + str(sess.run(accuracy, feed_dict={x: X_test, y_: y_test})))
print("Time: " + str(1000*(toc - tic)) + " ms")
