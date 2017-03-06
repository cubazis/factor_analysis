import numpy as np
import tensorflow as tf
import utils
BATCHSIZE = 10000
K = 3
MAX_ITER = 800
LR = 0.1

data = np.load('data2D.npy')
print(data.shape)
DIM = data.shape[1]


def DistFunc(X, Y):
  return -(tf.matmul(X, Y) * 2 - tf.reduce_sum(tf.square(tf.transpose(Y)), 0, keep_dims=True) - tf.reduce_sum(
      tf.square(X), 1, keep_dims=True))


def KmeansObjFunc(X, mu):

  dist = DistFunc(X, mu)
  label = tf.argmin(dist, axis=1)
  obj = tf.reduce_sum(dist[:, label])

  return obj

graph = tf.Graph()
with graph.as_default():
  inputPL = tf.placeholder(tf.float32, shape=(BATCHSIZE, DIM))

  ## Initialization
  weights = tf.Variable(tf.truncated_normal([DIM, K]) * 0.01)
  bias_pi = tf.Variable(tf.zeros([K]))
  bias_sigma = tf.Variable(tf.ones([K]) * (-5))
  ## transform the variables to meet the constrains
  Var = tf.exp(bias_sigma) + tf.constant(1e-8)
  logPi = utils.logsoftmax(tf.reshape(bias_pi, (1, K)))

  ## compute the log prob and posterior
  posterior, logProb = posteriorAndMariginalFunc(inputPL, weights, Var, logPi)

  optimizer = tf.train.AdamOptimizer(
      LR, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(-logProb)

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for i in range(MAX_ITER):
    _, output1, output2, mu, logVar, logPi = session.run(
        [optimizer, posterior, logProb, weights, bias_sigma, bias_pi],
        feed_dict={inputPL: data})
    if (i % 50) == 0:
      import pylab as plt
      plt.figure()
      plt.scatter(data[:, 0], data[:, 1], c=output1)
      plt.scatter(mu.T[:, 0], mu.T[:, 1], marker='s', c='c', s=50)
      plt.savefig('figures/f_%d.png' % (i))
      print(output2)
  print(output1)
  print(output1.sum(0))
  print(mu)
  print(np.exp(logVar))
  print(logPi)
