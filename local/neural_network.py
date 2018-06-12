import tensorflow as tf
import numpy as np
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def model(adj,res_connect, weights, X_train, y_train, X_test, y_test, prob = {}, optimizer="gradient_descent",learning_rate=0.5, batch_size=100, tests=3, num_epoch= 1000, config=""):

  dim_in = X_train.shape[1]
  dim_out = y_train.shape[1]
  n_samples = X_train.shape[0]
  first_level = []
  last_level = {}
  weight = {}
  bias = {}

  tf.reset_default_graph()

  # n = np.array([tf.Variable(0.) for _ in range(len(adj))])
  n = np.array([tf.Variable(0., name="Node") for _ in range(len(adj))])

  
  sess = tf.Session()

  x = tf.placeholder(tf.float32, [None, dim_in], name="x")
  y_ = tf.placeholder(tf.float32, [None, dim_out], name="labels")


  # initialization
  print('Initialization')

  last_level = set(adj.keys())

  with tf.name_scope("lattice"):
      for i in adj:
          if len(adj[i]) == 0:
              first_level.append(i)
          for j in adj[i]:
              last_level.discard(j)
              weight[str(i) + ',' + str(j)] = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1), name="W_lattice")
              # tf.summary.histogram("weights", weight[str(i) + ',' + str(j)])
          bias[str(i)] = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1), name="B_lattice")
          # tf.summary.histogram("biases", bias[str(i)])


  last_level = list(last_level)
  print(first_level)
  print(last_level)

  # weight['in'] = tf.Variable(tf.truncated_normal([dim_in,len(first_level)], stddev = 0.1))
  # bias['in'] = tf.Variable(tf.truncated_normal([len(first_level)], stddev = 0.1))

  # weight['out'] = tf.Variable(tf.truncated_normal([len(last_level), dim_out], stddev = 0.1))
  # bias['out'] = tf.Variable(tf.truncated_normal([dim_out], stddev = 0.1))

  for i in first_level:
      dev = np.random.rand(dim_in,1)/np.sqrt(dim_in)
      mean = np.zeros((dim_in,1))
      mean[weights[i]] = 4/np.sqrt(dim_in)
      weight['in,'+str(i)] = tf.Variable( mean+dev, dtype=tf.float32, name="W_in")
      # weight['in,'+str(i)] = tf.Variable(tf.truncated_normal([dim_in, 1], stddev=0.1), name="W_in")
      # tf.summary.histogram("weights", weight['in,'+str(i)])

  for i in last_level:
      weight[str(i)+',out'] = tf.Variable(tf.truncated_normal([1, dim_out], stddev=0.1), name="W_out")
      # tf.summary.histogram("weights", weight[str(i)+',out'])

  bias['out'] = tf.Variable(tf.truncated_normal([1,dim_out], stddev=0.1), name="B_out")
  # tf.summary.histogram("biases", bias['out'])

  with tf.name_scope("resnet"):
      for c in res_connect:
          temp = np.zeros(dim_out)
          temp[c] = 1
          weight['res_'+str(c)] = tf.constant(temp, dtype=tf.float32, name="W_res")
          # tf.summary.histogram("weights",  weight['res_'+str(c)])



  # network connecting
  print('Network connecting')

  for i in first_level:
      n[i] = tf.nn.relu(tf.matmul(x, weight['in,'+str(i)]) + bias[str(i)])

  def process(v):
      activation = tf.Variable(0.)
      if len(adj[v]) == 0:
          return
      for w in adj[v]:
          process(w)
          activation += weight[str(v) + ',' + str(w)] * n[w]
          # activation += tf.nn.dropout(weight[str(v) + ',' + str(w)],keep_prob=prob[str(w)+"_"+str(v)]) * n[w]
      n[v] = tf.nn.relu(activation + bias[str(v)])

  for v in last_level:
      process(v)

  y0 = bias['out']
  for i in last_level:
      y0 += n[i] * weight[str(i)+',out']


  for c in res_connect:
      activation = tf.Variable(0.)
      for i in res_connect[c]:
          activation += n[i]
      y0 += activation * weight['res_'+str(c)]



  # y = tf.nn.softmax(y0)


  # training
  print('Training')

  with tf.name_scope("cross_entropy"):
      # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
      #                                           tf.log(y), reduction_indices=[1]))
      cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y0))
      tf.summary.scalar("cross_entropy", cross_entropy)
  with tf.name_scope("train"):
    if optimizer=="gradient_descent":
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    elif optimizer == "adagrad":
      train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
    elif optimizer == "adam":
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y0, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  
  summ = tf.summary.merge_all()
      


  results = np.zeros((tests,))
  times = np.zeros((tests,))  #times in ms
  for test in range(tests):
      writer = tf.summary.FileWriter("tb_data/"+config+str(test))
      tic = time.time()
      init = tf.global_variables_initializer()
      sess = tf.Session()
      sess.run(init)
      # writer.add_graph(sess.graph)
      for i in range(num_epoch):
          indices = np.random.choice(n_samples, batch_size)
          x_batch, y_batch = X_train[indices], y_train[indices]
          if i % 5 == 0:
                [train_accuracy,s] = sess.run([accuracy,summ], feed_dict={x: x_batch, y_: y_batch})
                writer.add_summary(s, i)
          sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

      toc = time.time()

      results[test] = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
      times[test] = 1000*(toc - tic)


  # print('\n')
  # for i in range(tests):
  #     print("Test " + str(i+1) + " :  Accuracy = " + str(results[i]) + ", Time = " + str(times[i]))
  # print("\nMean value of accuracy: " + str(results.mean()))
  # print("Mean value of time: " + str(times.mean())+"\n")
  print(config)
  return results, times