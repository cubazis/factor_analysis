import math
import numpy as np
import tensorflow as tf
import utils as ut
from matplotlib.patches import Ellipse

BATCHSIZE = 10000
K = 30
MAX_ITER = 800
LR = 0.1
COLOR_LIST = ['r', 'g', 'b', 'y', 'm', 'k']
IS_VALID = False

# data = np.load('data/data2D.npy')
data = np.load('data/data100D.npy')

print(data.shape)
DIM = data.shape[1]
NUM_PTS = data.shape[0]
NUM_VALID = int(math.floor(NUM_PTS / 3.0))

if IS_VALID:
  npr = np.random.RandomState(1234)
  data_idx = npr.permutation(NUM_PTS)
  val_data = data[data_idx[:NUM_VALID]]
  data = data[data_idx[NUM_VALID:]]


def log_pdf_mix_gaussian(X, mu, sigma, log_pi):
  """ log pdf of mixture gaussian with covariance sigma * I

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
  log_joint_prob = log_likelihood
  log_likelihood = ut.reduce_logsumexp(log_likelihood, keep_dims=True)  # B x 1
  log_posterior = log_joint_prob - log_likelihood

  return tf.reduce_sum(log_likelihood), log_posterior


graph = tf.Graph()
with graph.as_default():
  inputPL = tf.placeholder(tf.float32, shape=(None, DIM))

  ## Initialization
  mu = tf.Variable(tf.truncated_normal([K, DIM]))
  sigma = tf.Variable(tf.truncated_normal([K, 1]))
  pi = tf.Variable(tf.random_uniform([K]))
  pi_normalize = tf.nn.softmax(pi)
  log_pi = ut.logsoftmax(pi)

  ## compute the log prob
  log_pdf, log_posterior = log_pdf_mix_gaussian(inputPL, mu, sigma, log_pi)

  optimizer = tf.train.AdamOptimizer(
      LR, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(-log_pdf)

train_loss = []
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for i in range(MAX_ITER):
    _, pdf_val, mu_np, sigma_np, pi_np, log_posterior_np = session.run(
        [optimizer, log_pdf, mu, sigma, pi_normalize, log_posterior],
        feed_dict={inputPL: data})
    train_loss += [-pdf_val]

    print('Iter {:07d}: log likelihood = {}'.format(i + 1, pdf_val))

    if ((i + 1) % 50) == 0 or i == 0:
      import pylab as plt
      fig = plt.figure()
      ax = fig.add_subplot(111)

      if not IS_VALID:
        # plt.scatter(data[:, 0], data[:, 1], c='c')
        # plt.scatter(
        #     mu_np[:, 0], mu_np[:, 1], marker='s', c=COLOR_LIST[:K], s=10)

        # # draw 1 std contour      
        # ells = [
        #     Ellipse(
        #         xy=mu_np[ii, :2],
        #         width=2 * np.sqrt(sigma_np[ii]**2),
        #         height=2 * np.sqrt(sigma_np[ii]**2)) for ii in xrange(K)
        # ]

        # for ii, ee in enumerate(ells):
        #   ax.add_artist(ee)
        #   ee.set_clip_box(ax.bbox)
        #   ee.set_alpha(0.5)
        #   ee.set_facecolor(COLOR_LIST[ii % len(COLOR_LIST)])
        for ii in xrange(K):
          idx = np.argmax(log_posterior_np, axis=1) == ii
          plt.scatter(data[idx, 0], data[idx, 1], c=COLOR_LIST[ii % len(COLOR_LIST)])
          plt.scatter(mu_np[ii, 0], mu_np[ii, 1], marker='s', c='c', s=80)

      else:
        loss_np = session.run([log_pdf], feed_dict={inputPL: val_data})

        for ii in xrange(K):
          idx = np.argmax(log_posterior_np, axis=1) == ii
          plt.scatter(data[idx, 0], data[idx, 1], c=COLOR_LIST[ii % len(COLOR_LIST)])
          plt.scatter(mu_np[ii, 0], mu_np[ii, 1], marker='s', c='c', s=80)

        plt.title("Validation loss = {}".format(-loss_np[0]))

      plt.savefig('figures/GMM_{:07d}.png'.format(i + 1))

  print("mu = {}".format(mu_np))
  print("sigma = {}".format(sigma_np))
  print("log_pi = {}".format(pi_np))

  # save loss function
  plt.figure()
  ax = plt.subplot(111)
  plt.plot(train_loss)
  ax.set_xlabel('Iteration')
  plt.title('Negative Log Likelihood')
  plt.savefig('figures/gmm_train_loss.png')
