import numpy as np
import tensorflow as tf
import math
import tensorboard as tb

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr, BATCH_SIZE, N_F):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.N_F = N_F

        self.s = tf.placeholder(tf.float32, [BATCH_SIZE, n_features], "state")
        self.a = tf.placeholder(tf.float32, BATCH_SIZE, "act")
        self.ap = tf.placeholder(tf.float32, BATCH_SIZE, "act_prob")
        self.adv = tf.placeholder(tf.float32, BATCH_SIZE, "adv")
        
        with tf.variable_scope('Actor'):
            mu_l1 = tf.layers.dense(
                inputs=self.s,
                units=64,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu_l1'
            )
            mu_l2 = tf.layers.dense(
                inputs=mu_l1,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu_l2'
            )
            self.mu = tf.layers.dense(
                inputs=mu_l2,
                units=1,    # output units
                activation=None,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu'
            )

            sig_l1 = tf.layers.dense(
                inputs=self.s,
                units=64,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='sig_l1'
            )
            sig_l2 = tf.layers.dense(
                inputs=sig_l1,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='sig_l2'
            )
            self.sig = tf.layers.dense(
                inputs=sig_l2,
                units=1,    # output units
                activation=tf.exp,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='sig'
            )

        with tf.variable_scope('ppo'):
            prob = 1/((self.sig + 1e-8) * tf.sqrt(2 * np.pi)) * tf.exp(-(self.a - self.mu)**2 / (2*(self.sig + 1e-8)**2))
            ratio = prob / self.ap
            pg_loss = self.adv * ratio
            clip_f = 0.2
            pg_loss2 = self.adv * tf.clip_by_value(ratio, 1.0-clip_f, 1.0+clip_f)
            self.pg_loss = tf.reduce_mean(tf.minimum(pg_loss, pg_loss2))
 
        with tf.variable_scope('train'):
            tf.summary.scalar('Vel Error', tf.reduce_mean(self.adv))
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.pg_loss) 

        with tf.variable_scope('parameterScope'):
            self.advAverage =  tf.reduce_mean(self.adv)

    def learn(self, mb_s, mb_a, mb_ap, mb_A):
        feed_dict = {self.s: mb_s, self.a: mb_a, self.ap: mb_ap, self.adv: mb_A}
        merged = tf.summary.merge_all()
        pg_loss, _, summary = self.sess.run([self.pg_loss, self.train_op, merged], feed_dict)
        
        return pg_loss, summary

    def choose_action(self, s):
        s_dummy = np.zeros((self.BATCH_SIZE, self.N_F))
        s_dummy[0,:] = s
        mu_,sig_ = self.sess.run([self.mu, self.sig], {self.s: s_dummy})
        sig_[0] = sig_[0] + 1e-8
        a = np.random.normal(mu_[0], sig_[0])
        ap = 1/(sig_[0] * math.sqrt(2 * np.pi)) * math.exp(-(a - mu_[0])**2 / (2*sig_[0]**2))
        a = a[0]
        ap = ap[0]
        return a,ap   # return 0-D numpy array variable

    def HER_ap_calculate(self, s, a):
        s_dummy = np.zeros((self.BATCH_SIZE, self.N_F))
        s_dummy[0,:] = s
        mu_,sig_ = self.sess.run([self.mu, self.sig], {self.s: s_dummy})
        sig_[0] = sig_[0] + 1e-8
        a_ = a
        ap = 1/(sig_[0] * math.sqrt(2 * np.pi)) * math.exp(-(a_ - mu_[0])**2 / (2*sig_[0]**2))
        ap = ap[0]
        return ap

    def parameterscope(self,mb_A):
        Adv = self.sess.run(self.advAverage,feed_dict={self.adv: mb_A})
        return Adv

