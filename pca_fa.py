import numpy as np
import tensorflow as tf

K = 1
MAX_ITER = 1000
LR = 1.0e-1
DIM = 3


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


def gen_hinton_data():

  seed = 1234
  npr = np.random.RandomState(seed)
  s = npr.randn(600).reshape(3, 200)
  A = np.array([[1, 0, 0], [1, 0.001, 0], [0, 0, 10]])
  x = np.dot(A, s)

  return x


def PCA(data):

  center_data = data - np.mean(data, axis=1, keepdims=True)
  cov_mat = np.dot(center_data, center_data.T) / (data.shape[1] - 1)
  _, P = np.linalg.eigh(cov_mat)

  return P, center_data


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
  data = gen_hinton_data()

  pca_pc, center_data = PCA(data)

  tf.global_variables_initializer().run()
  print('Initialized')
  session.run(mu_init_op, feed_dict={mu_init: np.mean(data.T, axis=0)})

  for i in range(MAX_ITER):
    _, log_p, W_np, mu_np, sigma_np = session.run(
        [optimizer, log_prob, W, mu, sigma], feed_dict={inputPL: data.T})

    print('Iter {:07d}: Negative Log likelihood = {:e}'.format(i + 1, -log_p))

  print("Sigma = {}".format(sigma_np))
  print("First component of PCA = {}".format(pca_pc[:, -1]))
  print("First component of FA = {}".format(W_np))
