import numpy as np
import tensorflow as tf
import utils as ut
from matplotlib.patches import Ellipse

BATCHSIZE = 10000
K = 3
MAX_ITER = 800
LR = 0.1
COLOR_LIST = ['r', 'g', 'b', 'y', 'm', 'k']

data = np.load('data/data2D.npy')
print(data.shape)
DIM = data.shape[1]


def log_mix_gaussian_pdf(X, mu, sigma, log_pi):
  """ pdf of mixture gaussian

    Args:
      X: B X D
      mu: K X D
      sigma: K X 1
      log_pi: K X 1

    Returns:
      log likelihood
  """

  Pi = tf.constant(float(np.pi))
  sigma_2 = tf.transpose(tf.square(sigma))  # K X 1
  diff = ut.pdist(X, mu)  # B X K

  log_likelihood = diff / sigma_2  # B X K
  log_likelihood += DIM * tf.log(2 * Pi)
  log_likelihood += DIM * tf.log(sigma_2)
  log_likelihood *= -0.5
  log_likelihood += tf.transpose(log_pi)
  log_likelihood = ut.reduce_logsumexp(log_likelihood)  # B x 1

  return tf.reduce_sum(log_likelihood)


graph = tf.Graph()
with graph.as_default():
  inputPL = tf.placeholder(tf.float32, shape=(BATCHSIZE, DIM))

  ## Initialization
  mu = tf.Variable(tf.truncated_normal([K, DIM]))
  sigma = tf.Variable(tf.truncated_normal([K, 1]))
  pi = tf.Variable(tf.random_uniform([K, 1]))
  log_pi = ut.logsoftmax(pi)

  ## compute the log prob
  log_pdf = log_mix_gaussian_pdf(inputPL, mu, sigma, log_pi)

  optimizer = tf.train.AdamOptimizer(
      LR, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(-log_pdf)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for i in range(MAX_ITER):
    _, pdf_val, mu_np, sigma_np, log_pi_np = session.run(
        [optimizer, log_pdf, mu, sigma, log_pi], feed_dict={inputPL: data})

    print('Iter {:07d}: log likelihood = {}'.format(i + 1, pdf_val))

    if (i % 50) == 0:
      import pylab as plt
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.scatter(data[:, 0], data[:, 1], c='c')
      plt.scatter(mu_np[:, 0], mu_np[:, 1], marker='s', c=COLOR_LIST[:K], s=10)

      # draw 1 std contour
      ells = [
          Ellipse(
              xy=mu_np[ii, :2],
              width=2 * np.sqrt(sigma_np[ii]**2),
              height=2 * np.sqrt(sigma_np[ii]**2)) for ii in xrange(K)
      ]

      for ii, ee in enumerate(ells):
        ax.add_artist(ee)
        ee.set_clip_box(ax.bbox)
        ee.set_alpha(0.5)
        ee.set_facecolor(COLOR_LIST[ii])

      plt.savefig('figures/GMM_{:07d}.png'.format(i + 1))
