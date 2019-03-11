import pybullet as p
import time
import pybullet_data
import numpy as np
import tensorflow as tf

class balldroid_sim():
    def __init__(self, step, isRender):
        
        self.step_size = step

        if isRender:
            self.render = p.GUI
        else:
            self.render = p.DIRECT

        self.observation_space = 12 

        physicsClient = p.connect(self.render)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        p.setGravity(0,0,-10)

        self.ballStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.ballStartPos = [0, 0, 0.18]

        planeId = p.loadURDF("simulator/custom_plane.urdf")
        self.boxId = p.loadURDF("simulator/10-balldroid.urdf",
                                self.ballStartPos, self.ballStartOrientation)

        self.weightVel = 0.5
        self.weightVib = 0.5
        self.inputVel = 0
        #self.velAmount = 10
        self.maxForce = 50 

        self.reset(0)

    def __del__(self):
        p.disconnect()

    def reset(self, target):
        
        p.resetBasePositionAndOrientation(self.boxId, self.ballStartPos,
                                          self.ballStartOrientation)
        p.resetJointState(self.boxId, 0, 0, 0)
        p.resetBaseVelocity(self.boxId, 0, 0)
        
        self.targetVel = target        
        self.inputVel = 0
        self.curr_vel = 0
        self.prev_action = 0
        
        s_, _, _  = self.step(np.random.randn())
        #return np.concatenate([np.zeros(self.observation_space - 1),np.array([self.targetVel])])
        return s_


    def step(self, action):

        #print("action :   ",action)
        self.inputVel = action
        for i in range(self.step_size):

            #add velocity
            p.setJointMotorControl2(self.boxId, 0, p.VELOCITY_CONTROL, 
                                    targetVelocity=self.inputVel, 
                                    force=self.maxForce)
            p.stepSimulation()

        # Get datas
        ballPos, ballOrn = p.getBasePositionAndOrientation(self.boxId)
        ballVelLinear, ballVelAngular = p.getBaseVelocity(self.boxId)
        jointState = p.getJointState(self.boxId, 0)
        robotPos, robotOri, _, _, _, _, _, robotAngVel \
                = p.getLinkState(self.boxId, 1, computeLinkVelocity=1)
        robotOriEuler = p.getEulerFromQuaternion(robotOri)

        #make data array
        data = np.array([ballVelLinear[0], ballVelLinear[1], ballVelLinear[2],
                         robotOriEuler[0], robotOriEuler[1], robotOriEuler[2],
                         robotAngVel[0],   robotAngVel[1],   robotAngVel[2],
                         jointState[1],    self.prev_action, self.targetVel])
        
        epsilon = 5e-2
        prev_error = self.curr_vel - self.targetVel
        curr_error = data[0] - self.targetVel
        prev_sign = np.sign(self.targetVel-self.curr_vel)
        curr_sign = np.sign(self.targetVel-data[0])
        reward = -0.5*(abs(prev_error) + abs(curr_error))

        
       
        self.prev_action = action
        
        done = (abs(self.targetVel - data[0]) < epsilon and \
                abs(self.targetVel - self.curr_vel) < epsilon)

        return data, reward, done  

