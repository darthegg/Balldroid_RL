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
        self.reward = tf.placeholder(tf.float32, BATCH_SIZE, "rew")
        self.v_esti = tf.placeholder(tf.float32, BATCH_SIZE, "estimation")
        self.v_rec = tf.placeholder(tf.float32, BATCH_SIZE, "record")
        self.adv = tf.placeholder(tf.float32, BATCH_SIZE, "adv")
                
        with tf.variable_scope('Actor'):
            mu_l1 = tf.layers.dense(
                inputs=self.s,
                units=128,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu_l1'
            )
            mu_l2 = tf.layers.dense(
                inputs=mu_l1,
                units=64,    # number of hidden units
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
            self.sig = tf.layers.dense(
                inputs=mu_l2,
                units=1,    # output units
                activation=tf.exp,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='sig'
            )
            
        with tf.variable_scope('ppo'):
            prob = 1/((self.sig + 1e-8) * tf.sqrt(2 * np.pi)) * tf.exp(-(self.a - self.mu)**2 / (2*(self.sig + 1e-8)**2))
            ratio = prob / self.ap
            pg_loss = -self.adv * ratio
            clip_f = 0.2
            pg_loss2 = -self.adv * tf.clip_by_value(ratio, 1.0-clip_f, 1.0+clip_f)
            
            self.pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss2))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.pg_loss)

    def learn(self, mb_s, mb_a, mb_ap, mb_reward, mb_v_esti, mb_v_rec, mb_adv):
        feed_dict = {self.s: mb_s, self.a: mb_a, self.ap: mb_ap, self.reward: mb_reward,
                     self.v_esti: mb_v_esti, self.v_rec: mb_v_rec, self.adv: mb_adv}
        pg_loss, _ = self.sess.run([self.pg_loss, self.train_op], feed_dict)
        
        return pg_loss

    def choose_action(self, s):
        s_dummy = np.zeros((self.BATCH_SIZE, self.N_F))
        s_dummy[0,:] = s
        mu_,sig_ = self.sess.run([self.mu, self.sig], {self.s: s_dummy})
        sig_[0] = sig_[0] + 1e-8
        a = np.random.normal(mu_[0], sig_[0])
        ap = 1/(sig_[0] * math.sqrt(2 * np.pi)) * math.exp(-(a - mu_[0])**2 / (2*sig_[0]**2))
        a = a[0]
        ap = ap[0]

        return a,ap


class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr, BATCH_SIZE, N_F):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.N_F = N_F

        self.s = tf.placeholder(tf.float32, [BATCH_SIZE, n_features], "state")
        self.a = tf.placeholder(tf.float32, BATCH_SIZE, "act")
        self.ap = tf.placeholder(tf.float32, BATCH_SIZE, "act_prob")
        self.reward = tf.placeholder(tf.float32, BATCH_SIZE, "rew")
        self.v_esti = tf.placeholder(tf.float32, BATCH_SIZE, "estimation")
        self.v_rec = tf.placeholder(tf.float32, BATCH_SIZE, "record")
        self.adv = tf.placeholder(tf.float32, BATCH_SIZE, "adv")
                
        with tf.variable_scope('Critic'):
            v_l1 = tf.layers.dense(
                inputs=self.s,
                units=128,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='v_l1'
            )
            v_l2 = tf.layers.dense(
                inputs=v_l1,
                units=64,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='v_l2'
            )            
            self.v = tf.layers.dense(
                inputs=v_l2,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='value'
            )

        with tf.variable_scope('ppo'):
            self.value_loss = tf.reduce_mean(tf.square(self.v_rec - self.v))
            
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.value_loss) 

    def learn(self, mb_s, mb_a, mb_ap, mb_reward, mb_v_esti, mb_v_rec, mb_adv):
        feed_dict = {self.s: mb_s, self.a: mb_a, self.ap: mb_ap, self.reward: mb_reward,
                     self.v_esti: mb_v_esti, self.v_rec: mb_v_rec, self.adv: mb_adv}
        value_loss,_ = self.sess.run([self.value_loss, self.train_op], feed_dict)
        
        return value_loss

    def cal_v(self, s):
        s_dummy = np.zeros((self.BATCH_SIZE, self.N_F))
        s_dummy[0,:] = s
        v_ = self.sess.run([self.v], {self.s: s_dummy})
        
        v = v_[0]
        return v   # return 0-D numpy array variable