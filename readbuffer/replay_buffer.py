import numpy as np

class ReadBuffer(object):
    def __init__(self,num_episode,num_step):
        self.num_episode = num_episode
        self.num_step = num_step
        self.ep_num = 0
        self.reset()
    
    def append(self, state, action, actionProb, reward, value, new_ep):
        self.state.append(state)
        self.action.append(action)
        self.ap.append(actionProb)

        self.rew_ep.append(reward)
        self.value_ep.append(value)
        self.done_ep.append(new_ep)

        return len(self.state)

    def add_gae(self, last_val, done_, gamma=0.99, lam=0.5):
        #self.A = np.asarray(self.A)
        #self.A = (self.A - self.A.mean()) / (self.A.std())
        rew = np.asarray(self.rew_ep)
        returns = np.zeros_like(rew)
        advs = np.zeros_like(rew)
        lastgaelam = 0
        for t in reversed(range(rew.shape[0])):
            if t == rew.shape[0] - 1:
                value_ = last_val
                term_mask = done_
            else:
                value_ = self.value_ep[t+1]
                term_mask = 1 - self.done_ep[t+1]
            delta = rew[t] + gamma * term_mask * value_ - self.value_ep[t]
            advs[t] = lastgaelam = delta + gamma * lam * term_mask * lastgaelam #???????
        A = (advs - advs.mean()) / advs.std()  # normalize
        ret = advs + np.asarray(self.value_ep)[:,0]  # Why not just advs? -> to compare with self.v

        if self.A == []:
            self.A = A
        else:
            self.A = np.concatenate([self.A, A], axis=0)
        if self.ret == []:
            self.ret = ret
        else:
            self.ret = np.concatenate([self.ret, ret], axis=0)
        self.rew += self.rew_ep
        self.value += self.value_ep
        self.done += self.done_ep

        self.rew_ep = []
        self.value_ep = []
        self.done_ep = []

        #print(self.A.shape, self.ret.shape, np.asarray(self.value).shape)

    def sample(self,batch_size):
        assert len(self.state) == self.num_episode * self.num_step
        #assert self.start < len(self.state) - batch_size
        sample_idx = self.index[self.start:self.start+batch_size]
        self.start += batch_size
         
        return (np.take(np.asarray(arr), sample_idx, axis=0) \
            for arr in (self.state, self.action, self.ap, 
                        self.A, self.ret, self.value))

    def reset(self):
        self.index = np.arange(self.num_episode*self.num_step)
        np.random.shuffle(self.index)
        self.state = []
        self.action = []
        self.ap = []

        self.rew = []
        self.value = []
        self.A = []
        self.ret = []
        self.done = []

        self.rew_ep = []
        self.value_ep = []
        self.done_ep = []
        self.start = 0

