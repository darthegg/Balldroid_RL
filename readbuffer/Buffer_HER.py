import numpy as np
import tensorflow as tf
import gym
from simulator import balldroid_reward
import math
import pickle
import os
import tensorboard as tb

class ReadBuffer(object):
    def __init__(self,num_episode,num_step):
        self.num_episode = num_episode
        self.num_step = num_step
        self.reset()
    
    def append(self,state,action,actionProb,advFncValue):
        if self.state.size == 0:
            self.state = np.append(self.state, state)
        else:
            self.state = np.vstack([self.state, state])
        self.action = np.append(self.action, action)
        self.ap = np.append(self.ap, actionProb)
        self.A = np.append(self.A, advFncValue)
        return len(self.state)

    def sample(self,batch_size):
        assert len(self.action) == self.num_episode * self.num_step *(self.numReplay + 1)
        assert self.start < len(self.state) - batch_size
        sample_idx = self.index[self.start:self.start+batch_size]
        self.start += batch_size

        return (np.take(arr, sample_idx, axis=0) for arr in (self.state, self.action, self.ap, self.A))

    def reset(self):
        self.index = np.arange(self.num_episode*self.num_step)
        np.random.shuffle(self.index)
        self.state = np.array([])
        self.action = np.array([])
        self.ap = np.array([])
        self.A = np.array([])
        self.start = 0
 
    def HER_init(self,targetRange,numReplay,makeReplayFlag):
        self.replayFlag = makeReplayFlag
        if makeReplayFlag:
            self.numReplay = numReplay
            self.replayTarget = np.random.rand(numReplay)*(targetRange[1] - targetRange[0]) + targetRange[0]
        else:
            self.numReplay = 0
    '''
    def makeReplay(self, actor): # Replay must be generated between each episodes
        if self.replayFlag:
            state_ = np.ndarray.tolist(stateArr[-self.num_step:])
            action_ = np.ndarray.tolist(actionArr[-self.num_step:])
            
            for replay_t in self.replayTarget:
                r = 0
                for i in range(self.num_step):
                    if i == self.num_step - 1:
                        i = i - 1
                    state_[i][-1] = replay_t
                    ap_ = actor.HER_ap_calculate(state_[i],action_[i])
                    r_ = (replay_t - state_[i+1][0])**2
                    adv_ = r_ - r
                    r = r_    
                    self.append(state_[i],action_[i],ap_,adv_)
        else:
            pass  
    '''