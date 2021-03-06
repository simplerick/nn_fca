
from data_loader import *
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


X, y, object_labels, attribute_labels = get_mammographic_masses()[:4]
y_cl = one_hot(y, n_classes=2)
X_train, y_train, X_test, y_test = train_test_split(
    X, y_cl, tp=0.8)


dim_in = X_train.shape[1]
dim_out = y_train.shape[1]
batch_size = 100
n_samples = X_train.shape[0]
num_epoch = 300
tests = 3
lr = 0.003

layer_dims = [dim_in, 20,20,20,20, dim_out]

x = tf.placeholder(tf.float32, [None,dim_in])
y_ = tf.placeholder(tf.float32, [None,dim_out])


W = []
b = []
h = []




for i in range(1,len(layer_dims)):
    W.append(tf.Variable(tf.truncated_normal([layer_dims[i-1],layer_dims[i]], stddev = np.sqrt(2/layer_dims[i-1]))))
    b.append(tf.Variable(tf.truncated_normal([layer_dims[i]], stddev = 0.1)))

h.append(x)
for i in range(1,len(layer_dims)-1):
    h.append(tf.nn.leaky_relu(tf.matmul(h[i-1],W[i-1])+b[i-1]))
    # if i < 3:
    #     h.append(tf.nn.relu(tf.matmul(h[i-1],W[i-1])+b[i-1]))
    # else:
    #     h.append(tf.nn.relu(tf.matmul(h[i-1],W[i-1])+b[i-1]+h[i-2]))
# h[3] = tf.nn.dropout(h[3],keep_prob=0.5)


y = tf.matmul(h[-1],W[-1])+b[-1]


with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

results = np.zeros((tests,))

for test in range(tests):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(int(num_epoch*n_samples/batch_size)):
        indices = np.random.choice(n_samples, batch_size)
        x_batch, y_batch = X_train[indices], y_train[indices]
        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    results[test] = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
    print(results[test])


for i in range(len(results)):
    print("test1: %s, accuracy: %s" %(i+1, results[i]))

print('\n')
print(np.mean(results))

f = open('data/dense_nn_perfomance_mam', 'a')
l = ",".join(map(str,layer_dims[1:-1]))
l = l+(10-len(l))*" "
l += ",".join(list(map(lambda x: "%.2f" % x,results))) 
l += "  %.4f" % results.mean() + "\n"
f.write(l)
f.close()