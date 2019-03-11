import numpy as np
import tensorflow as tf
from simulator import balldroid_light
from network import A2C_model
from readbuffer import Buffer_GAE
import math
import pickle
import os
import tensorboard as tb

np.random.seed(1) # random geneator start point initialize
tf.set_random_seed(1)  # reproducible (do the same learning process as before try)

# Path
SAVE_FILE_PATH = "./modelsave/model_vel_noObs_190221.ckpt"
LOG_FILE_PATH = './logsave/log_vel_noObs_190304.p'

os.makedirs("/tmp", exist_ok=True)
RESUME = True
RENDER = False

# Hyper parameters
MAX_UPDATE = 5000
MAX_EPISODE = 8
MAX_EP_STEPS = 700   # MAX_EP_STEPS/240 = realtime sec
MINI_EP_STEPS = 50
BATCH_SIZE = 64
EPOCH = 70
LR_A = 0.001    # learning rate for actor
LR_C = 0.01
GAMMA = 0.99
env = balldroid_light.balldroid_sim(1,RENDER)
N_F = env.observation_space #env.observation_space.shape[0]
N_A = 3 #env.action_space.n, meaningless
MakeReplayFlag = False
NumReplay = 4

# Environment setting
obstaclePos = [-20, -22, -24]
targetRange = [0.49,0.5]

# Initialize
sess = tf.Session()
actor = A2C_model.Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A, BATCH_SIZE=BATCH_SIZE, N_F=N_F)
critic = A2C_model.Critic(sess, n_features=N_F, n_actions=N_A, lr=LR_C, BATCH_SIZE=BATCH_SIZE, N_F=N_F)
rb = Buffer_GAE.ReadBuffer(MAX_EPISODE,MAX_EP_STEPS,MINI_EP_STEPS)
saver = tf.train.Saver()

# Save Path
os.makedirs('/logs', exist_ok=True)
if RESUME:
    saver.restore(sess, SAVE_FILE_PATH)
else:
    sess.run(tf.global_variables_initializer())

# For logging & display
datalog = np.zeros((MAX_EP_STEPS,2))
Alog = np.zeros(MAX_EP_STEPS)
stackdatalog = np.zeros((MAX_EP_STEPS,1))
track_loss = np.zeros((EPOCH,3))
for i in range(MAX_EP_STEPS):
    stackdatalog[i,0] = i
for i in range(EPOCH):
    track_loss[i,0] = i

# Loop
for update in range(MAX_UPDATE):
    rb.reset()
    print('Update No. ',update)
    rb.HER_init(targetRange,NumReplay,MakeReplayFlag)

    for i_episode in range(MAX_EPISODE):
        targetVel = np.random.rand()*(targetRange[1]-targetRange[0]) + targetRange[0]
        s = env.reset(targetVel, obstaclePos)
        notValue = -(targetVel**2)
        t = 0
        datalog[:,:] = 0    
        track_loss[:,:] = 0
    
        for t in range(MAX_EP_STEPS):
            a,ap = actor.choose_action(s)
            v_estimate = critic.cal_v(s)
            s_, notValue_, done = env.step(a)
            cost = s_[3]*0.1
            reward = notValue_ - notValue - cost
            rb.append(s,a,ap,reward,v_estimate,0.0,0.0)
            s = s_
            notValue = notValue_
            datalog[t,0] = s_[0] # logging velocity of ball
            datalog[t,1] = s_[3]
        
        rb.calculate_v_rec_adv(GAMMA)
   
        if i_episode == 0:
            stackdatalog = np.concatenate((stackdatalog,datalog),axis = 1)

        velAverage = np.average(datalog[-200:,0])
        print("episode:", i_episode)
        print("Target Vel:",targetVel,"  Actual Vel Average:",velAverage)

    for i in range(EPOCH): # MAX_EPISODE*MAX_EP_STEPS // BATCH_SIZE
        mb_s,mb_a,mb_ap,mb_reward,mb_v_estimate,mb_v_record,mb_adv = rb.sample(BATCH_SIZE)
        pg_loss = actor.learn(mb_s, mb_a, mb_ap, mb_reward, mb_v_estimate, mb_v_record,mb_adv)
        value_loss = critic.learn(mb_s, mb_a, mb_ap, mb_reward, mb_v_estimate, mb_v_record,mb_adv)
        track_loss[i,1] = pg_loss
        track_loss[i,2] = value_loss
        
    save_path = saver.save(sess, SAVE_FILE_PATH)
    pickle.dump(stackdatalog, open(LOG_FILE_PATH, 'wb'))
    lossAverage = np.average(track_loss[:,1])
    vlossAverage = np.average(track_loss[:,2])    
    print('pg_loss: ', lossAverage, 'value_loss: ',vlossAverage)

