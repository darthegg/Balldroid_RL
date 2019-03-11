import numpy as np
import math

class ReadBuffer(object):
    def __init__(self,num_episode,num_step,mini_step):
        self.num_episode = num_episode
        self.num_step = num_step
        self.mini_step = mini_step
        self.reset()
    

    def append(self,state,action,actionProb,reward,v_estimate,v_record,adv):
        if self.state.size == 0:
            self.state = np.append(self.state, state)
        else:
            self.state = np.vstack([self.state, state])
        self.action = np.append(self.action, action)
        self.ap = np.append(self.ap, actionProb)
        self.reward = np.append(self.reward, reward)
        self.v_estimate = np.append(self.v_estimate, v_estimate)
        self.v_record = np.append(self.v_record, v_record)
        self.adv = np.append(self.adv, adv)
                
        return len(self.state)


    def calculate_v_rec_adv(self,gamma):
        stacklen = self.num_step-self.mini_step
        vlog = np.zeros(stacklen)
        A = np.zeros(stacklen)

        for t in range(stacklen):
            vlog[t] = self.v_estimate[-stacklen+t]
            for i in range(self.mini_step):
                vlog[t] = self.reward[-stacklen+t-i-1] + gamma*vlog[t]

        # Slice
        self.state = self.state[:-self.mini_step]
        self.action = self.action[:-self.mini_step]
        self.ap = self.ap[:-self.mini_step]
        self.reward = self.reward[:-self.mini_step]
        self.v_estimate = self.v_estimate[:-self.mini_step]
        self.v_record = self.v_record[:-self.mini_step] 
        self.adv = self.adv[:-self.mini_step]

        self.v_record[-stacklen:] = vlog

        for t in range(stacklen):
            A[t] = self.v_record[-stacklen+t] - self.v_estimate[-stacklen+t]
        
        self.adv[-stacklen:] = (A - A.mean()) / A.std()
        #self.adv[-stacklen:] = A[t]


    def sample(self,batch_size):
        assert len(self.action) == self.num_episode * (self.num_step-self.mini_step) * (self.numReplay + 1)
        assert self.start < len(self.state) - batch_size
        sample_idx = self.index[self.start:self.start+batch_size]
        self.start += batch_size

        return (np.take(arr, sample_idx, axis=0) for arr in (self.state, 
                                                             self.action, 
                                                             self.ap, 
                                                             self.reward, 
                                                             self.v_estimate, 
                                                             self.v_record,
                                                             self.adv))

    def HER_init(self,targetRange,numReplay,makeReplayFlag):
        self.replayFlag = makeReplayFlag
        if makeReplayFlag:
            self.numReplay = numReplay
            self.replayTarget = np.random.rand(numReplay)*(targetRange[1] - targetRange[0]) + targetRange[0]
        else:
            self.numReplay = 0

    def reset(self):
        self.index = np.arange(self.num_episode*(self.num_step-self.mini_step))
        np.random.shuffle(self.index)
        self.state = np.array([])
        self.action = np.array([])
        self.ap = np.array([])
        self.reward = np.array([])
        self.v_estimate = np.array([])
        self.v_record = np.array([])
        self.adv = np.array([])
        self.start = 0
