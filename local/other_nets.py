
from data_loader import *
import tensorflow as tf


X, y, object_labels, attribute_labels = get_titanic()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(
    X, y_cl, tp=0.6, vp=0.2)


dim_in = X_train.shape[1]
dim_out = y_train.shape[1]
batch_size = 100
n_samples = X_train.shape[0]
num_epoch = 1000
tests = 10

layer_dims = [dim_in, 15,15,15, dim_out]

x = tf.placeholder(tf.float32, [None,dim_in])
y_ = tf.placeholder(tf.float32, [None,dim_out])


W = []
b = []
h = []

w_out = np.ones((layer_dims[-2],dim_out))


for i in range(1,len(layer_dims)):
    W.append(tf.Variable(tf.truncated_normal([layer_dims[i-1],layer_dims[i]], stddev = 0.1)))
    b.append(tf.Variable(tf.truncated_normal([layer_dims[i]], stddev = 0.1)))

h.append(x)
for i in range(1,len(layer_dims)-1):
    # h.append(tf.nn.relu(tf.matmul(h[i-1],W[i-1])+b[i-1]))
    if i < 3:
        h.append(tf.nn.relu(tf.matmul(h[i-1],W[i-1])+b[i-1]))
    else:
        h.append(tf.nn.relu(tf.matmul(h[i-1],W[i-1])+b[i-1]+h[i-2]))


y = tf.nn.softmax(tf.matmul(h[-1],W[-1])+b[-1])


with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y), reduction_indices=[1]))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

results = np.zeros((tests,))

for test in range(tests):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(num_epoch):
        indices = np.random.choice(n_samples, batch_size)
        x_batch, y_batch = X_train[indices], y_train[indices]
        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    results[test] = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})


for i in range(len(results)):
    print("iteration: %s, accuracy: %s" %(i+1, results[i]))

print('\n')
print(np.mean(results))

f = open('dense_nn_perfomance', 'a')
l = ",".join(map(str,layer_dims[1:-1]))
l = l+(10-len(l))*" "
l += ",".join(list(map(lambda x: "%.2f" % x,results))) 
l += "  %.4f" % results.mean() + "\n"
f.write(l)
f.close()