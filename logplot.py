import numpy as np
import matplotlib.pyplot as plt
import pickle

log = pickle.load(open('./logsave/log_vel_noObs_190304.p','rb'))
#log = pickle.load(open('./logsave/log_curri_190310.p','rb'))
#log = pickle.load(open('./logsave/log_pid_190310.p','rb'))

'''
for i in range(100):
    plt.plot(log[:,0],log[:,i])

    plt.show()
'''


for i in range(10):
    plt.plot(log[:,0],log[:,-i])
    #plt.plot(log[:])
    '''
    axes = plt.gca()
    axes.set_xlim([0, 256])
    axes.set_ylim([-0.5, 1.0])
	'''
    plt.show()

