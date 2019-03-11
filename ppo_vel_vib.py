import numpy as np
import tensorflow as tf
import gym
import math
import pickle
import os
import tensorboard as tb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from simulator import balldroid_vel
from readbuffer import replay_buffer
from network import models

# Random or fix seed
# np.random.seed(1) # random geneator start point initialize
# tf.set_random_seed(1)  # reproducible (do the same learning process as before try)

# Init hyperparams, agents, etc.
LOG_FILE_PATH = './logsave/log_vel_noObs_190307.p'
SAVE_FILE_PATH = "/tmp/model.ckpt"
os.makedirs("/tmp", exist_ok=True)
#RESUME = True and os.path.exists(SAVE_FILE_PATH)
RESUME = False

MAX_UPDATE = 1000
MAX_EPISODE = 8

MAX_EP_STEPS = 256  # MAX_EP_STEPS/240 = realtime sec
BATCH_SIZE = 128
EPOCH = (MAX_EP_STEPS*MAX_EPISODE) // BATCH_SIZE #// 4 

RENDER = False   # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

n_stack = 2
n_skip = 4

env = balldroid_vel.balldroid_sim(n_skip, RENDER)

N_F = env.observation_space
N_A = 41

sess = tf.Session()
actor = models.Actor(sess, BATCH_SIZE, n_features=N_F*n_stack, n_actions=N_A, 
              lr=LR_A, disc=True)
saver = tf.train.Saver()

merged = tf.summary.merge_all()
os.makedirs('logs', exist_ok=True)
train_writer = tf.summary.FileWriter('./logs',sess.graph)

if RESUME:
    saver.restore(sess, SAVE_FILE_PATH)
else:
    sess.run(tf.global_variables_initializer())

rb = replay_buffer.ReadBuffer(MAX_EPISODE,MAX_EP_STEPS)

# For logging purposes
datalog = np.zeros((MAX_EP_STEPS,1))
stackdatalog = np.zeros((MAX_EP_STEPS,1))
track_pg_loss = np.zeros((EPOCH,2))

for i in range(MAX_EP_STEPS):
    stackdatalog[i,0] = i

for i in range(EPOCH):
    track_pg_loss[i,0] = i

velAverage = 0
targetVel = 0.5
state_buffer = None

# Run the training process
for update in range(MAX_UPDATE):
    rb.reset()

    diff_sum = 0
    r_sum = 0
    for i_episode in range(MAX_EPISODE):
        #targetVel = (np.random.rand()-0.5)*4.0
        s = env.reset(targetVel)
        r = 0
        t = 0
        datalog[:,:] = 0    
        track_pg_loss[:,:] = 0     
        mu_pool = []
        vel_pool = []
        done = False
        for t in range(MAX_EP_STEPS):
            ep_start = (t==0)
            if state_buffer is None:
                state_buffer = np.concatenate([s for _ in range(n_stack)])
            else:
                state_buffer = np.concatenate([state_buffer[N_F:], s])
            a, ap, v = actor.choose_action(state_buffer)
            s_, r_, done_ = env.step(a)
            r_sum += r_

            rb.append(state_buffer.copy(), a, ap, r_, v, done)
            datalog[t,0] = s_[0] # logging velocity of ball
            r = r_    
            s = s_
            done = done_

            mu_pool.append(a)
            vel_pool.append(s[0])

        state_buffer = np.concatenate([state_buffer[N_F:], s_])
        _, _, last_val = actor.choose_action(state_buffer)
        rb.add_gae(last_val, done_)

        if i_episode == 0:
            stackdatalog = np.concatenate((stackdatalog,datalog),axis = 1)

        # For debugging with plot
        if update < 0:
            plt.figure()
            #plt.plot(mu_pool)
            plt.plot(vel_pool)
            plt.show()

        velAverage = np.average(datalog[-20:,0])
        diff_sum += abs(targetVel - velAverage)

    val_loss_sum = 0
    for i in range(EPOCH): # MAX_EPISODE*MAX_EP_STEP // BATCH_SIZE
        mb_s, mb_a, mb_ap, mb_A, mb_ret, mb_v = rb.sample(BATCH_SIZE)
        pg_loss, val_loss, summary = actor.learn(mb_s, mb_a, mb_ap, 
                                                 mb_A, mb_ret, mb_v)  
        val_loss_sum += val_loss
        #track_pg_loss.append(pg_loss)
        if i==0:
            train_writer.add_summary(summary, update)
        track_pg_loss[i,1] = pg_loss

    # Print for debugging
    print("UPDATE {}".format(update))
    print("avg diff: ", diff_sum / MAX_EPISODE)
    print("avg adv: ", r_sum / (MAX_EPISODE * MAX_EPISODE))
    print("val loss: ", val_loss_sum / EPOCH)
    print()
    
    
    save_path = saver.save(sess, SAVE_FILE_PATH)
    lossAverage = np.average(track_pg_loss[:,1])
    pickle.dump(stackdatalog, open(LOG_FILE_PATH, 'wb'))


'''
TODO:
    - test hyperparameters (num_update, batch_size (32, 64, 128), epoch_size(fixed, full), num_episode(8, 16), max_step(~), lr(0.1~0.0001))
    - print status per update (statistics)
    - save checkpoint) - done
    - save first trajectory of each update (for plotting trajectory) - done
    - save loss data for plotting loss graph - ??? pg_loss size???
    - generate plots! - done

    - standardize adv_values
    - normalize(standardize inputs)
    - ** multiple target velocity ** (conditional input) HER
    - advanced reward function (fluctuation, etc)
'''
 