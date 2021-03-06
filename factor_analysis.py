import numpy as np
import tensorflow as tf
import utils
BATCHSIZE = 10000
K = 4
MAX_ITER = 800
LR = 1.0e-1

data = np.load("data/tinymnist.npz")
print(data["x"].shape)
DIM = data["x"].shape[1]


def log_pdf_factor_analysis(X, W, mu, sigma):
  """ log pdf of factor analysis

    Args:
      X: B X D
      W: D X K
      mu: D X 1
      sigma: D X 1

    Returns:
      log likelihood
  """

  Pi = tf.constant(float(np.pi))
  diff_vec = X - mu
  sigma_2 = tf.square(sigma)
  # phi = tf.eye(K) * sigma_2
  # M = tf.matmul(W, W, transpose_a=True) + phi
  # using Sherman-Morrison-Woodbury formula to compute the inverse
  # inv_M = tf.matrix_inverse(M)
  # inv_cov = tf.eye(DIM) / sigma_2 + tf.matmul(
  #     tf.matmul(W, inv_M), W, transpose_b=True) / sigma_2

  # using Sylvester's determinant identity to compute log determinant
  # implementation 1: directly compute determinant
  # log_det = tf.log(tf.matrix_determinant(M)) + 2.0 * (DIM - K) * tf.log(sigma)

  # implementation 2: using Cholesky decomposition
  # log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M)))) + 2.0 * (
  #     DIM - K) * tf.log(sigma)

  # phi = tf.eye(DIM) * sigma_2
  phi = tf.diag(sigma_2)
  M = phi + tf.matmul(W, W, transpose_b=True)
  inv_cov = tf.matrix_inverse(M)
  log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M))))

  log_likelihood = tf.matmul(
      tf.matmul(diff_vec, inv_cov), diff_vec, transpose_b=True)
  log_likelihood = tf.diag_part(log_likelihood)
  log_likelihood += DIM * tf.log(2 * Pi)
  log_likelihood += log_det
  log_likelihood = tf.reduce_sum(log_likelihood) * (-0.5)

  return log_likelihood


graph = tf.Graph()
with graph.as_default():
  inputPL = tf.placeholder(tf.float32, shape=(None, DIM))
  mu_init = tf.placeholder(tf.float32, shape=(DIM))

  ## Initialization
  W = tf.Variable(tf.truncated_normal([DIM, K]) * 0.1)
  mu = tf.Variable(tf.zeros([DIM]))
  sigma = tf.Variable(tf.ones([DIM]))

  mu_init_op = tf.assign(mu, mu_init)

  ## compute the log prob and posterior
  log_prob = log_pdf_factor_analysis(inputPL, W, mu, sigma)

  optimizer = tf.train.AdamOptimizer(
      LR, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(-log_prob)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  session.run(mu_init_op, feed_dict={mu_init: np.mean(data["x"], axis=0)})

  for i in range(MAX_ITER):
    _, log_p, W_np, mu_np, sigma_np = session.run(
        [optimizer, log_prob, W, mu, sigma], feed_dict={inputPL: data["x"]})

    print('Iter {:07d}: Log likelihood = {:e}'.format(i + 1, log_p))
    # print('sigma = {}'.format(sigma_np))

    if ((i + 1) % 50) == 0 or i == 0:
      # project into latent space
      # data_proj = np.dot((data - mu_np), W_np)
      import pylab as plt
      for kk in xrange(K):
        plt.figure()
        # plt.scatter(data[:, 0], data[:, 1], c='b')
        # plt.scatter(data_proj[:, 0], data_proj[:, 1], marker='s', c='g', s=50)
        plt.imshow(W_np[:, kk].reshape(8, 8), cmap="gray")
        plt.savefig(
            'figures/factor_analysis_{:07d}_{:02d}.png'.format(i + 1, kk + 1))
        plt.close()

  log_p_valid = session.run(log_prob, feed_dict={inputPL: data["x_valid"]})
  log_p_test = session.run(log_prob, feed_dict={inputPL: data["x_test"]})

  print("Validation log prob = {}".format(log_p_valid))
  print("Testing log prob = {}".format(log_p_test))
