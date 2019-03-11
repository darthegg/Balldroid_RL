import tensorflow as tf 
import numpy as np 
import math 

class Actor(object):
    def __init__(self, sess, batch_size, n_features, n_actions, lr, disc=False, a_max=10):
        self.sess = sess
        if not disc:
            n_actions = 2 
        self.disc = disc
        self.n_features = n_features
        self.n_actions = n_actions
        self.a_max = a_max
        self.batch_size = batch_size
        self.s = tf.placeholder(tf.float32, [batch_size, n_features], "state")
        self.a = tf.placeholder(tf.float32, batch_size, "act")
        self.ap = tf.placeholder(tf.float32, batch_size, "act_prob")
        self.adv = tf.placeholder(tf.float32, batch_size, "adv")
        self.ret = tf.placeholder(tf.float32, batch_size, "return")
        self.old_v = tf.placeholder(tf.float32, [batch_size,1], "old_v")
        with tf.variable_scope('Actor'):
            mu_l1 = tf.layers.dense(
                inputs=self.s,
                units=128,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1), # weights
                bias_initializer=tf.constant_initializer(0.1), # biases
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

            mu_l3 = tf.layers.dense(
                inputs=mu_l2,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                name='mu_l3'
            ) 

            self.mu_sig = tf.layers.dense(
                inputs=mu_l2,
                units=n_actions,    # output units
                activation=None,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu'
            )
            
            self.v = tf.layers.dense(
                inputs=mu_l2,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='value'
            )
            
            if disc:
                self.a_prob = tf.nn.softmax(self.mu_sig)
            else:
                self.mu = self.mu_sig[:,:self.mu_sig.shape[-1]//2]
                self.var = tf.exp(self.mu_sig[:,self.mu_sig.shape[-1]//2:])
                self.sig = tf.sqrt(self.var)

        if disc:
            a_idx = self.a / (self.a_max/(self.n_actions//2)) + \
                    self.n_actions//2
            a_idx_int = tf.cast(a_idx, tf.int32)
            prob = tf.gather(self.a_prob, a_idx_int, axis=1)
        else:
            prob = 1/(self.sig * tf.sqrt(2 * np.pi)) * tf.exp(-(self.a - self.mu)**2 / (2*self.sig**2))
        ratio = prob / self.ap
        pg_loss = -self.adv * ratio

        clip_f = 0.2
        pg_loss2 = -self.adv * tf.clip_by_value(ratio, 1.0-clip_f, 1.0+clip_f)
       
        v_clipped = self.old_v + tf.clip_by_value(self.v - self.old_v, -clip_f, clip_f)
        v_loss1 = tf.square(self.v - self.ret)
        v_loss2 = tf.square(v_clipped - self.ret)
        self.value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1,  v_loss2))
        #self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.v - self.ret))
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss2))
        #self.energy = tf.square(self.mu - self.s[:,-2])
        #with tf.variable_scope('train'):
        tf.summary.scalar('Vel Error', tf.reduce_mean(self.adv))
        optimizer = tf.train.AdamOptimizer(lr)
        #optimizer = tf.train.RMSPropOptimizer(lr)
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        all_loss = self.pg_loss + 0.5 * self.value_loss
        if disc:
            entropy = tf.reduce_mean(tf.reduce_sum(-self.a_prob*tf.log(self.a_prob), axis=1))
            all_loss = all_loss - 0.01 * entropy
        self.train_op = optimizer.minimize(all_loss) #+ 0.01 * self.energy)



    def learn(self, mb_s, mb_a, mb_ap, mb_A, mb_ret, mb_v):
        feed_dict = {self.s: mb_s, self.a: mb_a, self.ap: mb_ap, 
                self.adv: mb_A, self.ret: mb_ret, self.old_v: mb_v}
        merged = tf.summary.merge_all()
        pg_loss, val_loss, _, summary = self.sess.run([
                                        self.pg_loss, self.value_loss, 
                                        self.train_op, merged], feed_dict)
        
        return pg_loss, val_loss, summary


    def choose_action(self, s):
        s_dummy = np.zeros((self.batch_size, self.n_features))
        s_dummy[0,:] = s
        if self.disc:
            a_prob, v = self.sess.run([self.a_prob, self.v], {self.s: s_dummy})
            a_prob = a_prob[0]
            a = np.random.multinomial(1, a_prob) # error
            a = np.argmax(a)
            ap = a_prob[a]
            a = (a - self.n_actions//2) * (self.a_max / (self.n_actions//2))
        else:
            mu_, sig_, v = \
                    self.sess.run([self.mu, self.sig, self.v], {self.s: s_dummy})
            mu_ = mu_[0]
            sig_ = sig_[0]
            try:
                a = np.random.normal(mu_, sig_)
            except:
                #print(mu_, sig_)
                pass
            ap = 1/(sig_ * math.sqrt(2 * np.pi)) * math.exp(-(a - mu_)**2 / (2*sig_**2))
            a = a[0]
            ap = ap[0]
        v = v[0]
        
        return a,ap,v   # return 0-D numpy array variable

