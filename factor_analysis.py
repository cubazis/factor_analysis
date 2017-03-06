import numpy as np
import tensorflow as tf
import utils
BATCHSIZE = 10000
K = 3
MAX_ITER = 800
LR = 1.0e-1

data = np.load('data100D.npy')
print(data.shape)
DIM = data.shape[1]


def multiGaussPDF(X, W, mu, sigma):
    """
    Args:
        X: B X D
        W: D X K
        mu: 1 X D
        sigma: K X 1
    """

    Pi = tf.constant(float(np.pi))
    diff_vec = X - mu
    sigma_2 = tf.square(sigma)
    phi = tf.eye(K) * sigma_2
    M = tf.matmul(W, W, transpose_a=True) + phi     
    # using Sherman-Morrison-Woodbury formula to compute the inverse
    inv_M = tf.matrix_inverse(M)
    inv_cov = tf.eye(DIM) / sigma + tf.matmul(tf.matmul(W, inv_M), W, transpose_b=True) / sigma_2
    
    # using Sylvester's determinant identity to compute log determinant
    # implementation 1: directly compute determinant
    log_det = tf.log(tf.matrix_determinant(M)) 

    # implementation 2: using Cholesky decomposition
    log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M))))

    log_likelihood = tf.matmul(tf.matmul(diff_vec, inv_cov), diff_vec, transpose_b=True)
    log_likelihood = tf.diag_part(log_likelihood)
    log_likelihood += DIM * tf.log(2*Pi)
    log_likelihood += log_det
    log_likelihood = tf.reduce_sum(log_likelihood) * (-0.5)

    return log_likelihood


def logLikelihoodFunc(X, W, mu, sigma):

    log_prob = multiGaussPDF(X, W, mu, sigma)

    return log_prob


graph = tf.Graph()
with graph.as_default():
    inputPL = tf.placeholder(tf.float32, shape=(BATCHSIZE, DIM))
    
    ## Initialization
    W = tf.Variable(tf.truncated_normal([DIM, K])*0.1)
    mu = tf.Variable(tf.zeros([DIM]))
    sigma = tf.Variable(tf.ones([1])*1.0e-0)

    ## compute the log prob and posterior
    log_prob = logLikelihoodFunc(inputPL, W, mu, sigma)
    
    optimizer = tf.train.AdamOptimizer(LR, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(-log_prob)
    

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for i in range(MAX_ITER):
        _, log_p, W_np, mu_np, sigma_np, cov_np = session.run([optimizer, log_prob, W, mu, sigma, cov], feed_dict={inputPL:data})
        
        # print('=' * 80)
        # print('cov = {}'.format(cov_np))
        print('=' * 80)
        print('sigma = {}'.format(sigma_np))
        print('=' * 80)
        print('Iter {:07d}: Log likelihood = {:e}'.format(i+1, log_p))

        if (i % 50) == 0:
            import pylab as plt
            plt.figure()
            plt.scatter(data[:,0], data[:,1], c='c')
            # plt.scatter(mu[:,0], mu.T[:,1], marker='s', c='c', s=50)
            plt.savefig('figures/f_%d.png'%(i))
            