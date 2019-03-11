import pybullet as p
import time
import pybullet_data
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as et
import math

class balldroid_sim():
    def __init__(self, step, isRender):
        
        self.step_size = step

        if isRender:
            self.render = p.GUI
        else:
            self.render = p.DIRECT

        self.observation_space = 5

        physicsClient = p.connect(self.render)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)

        self.ballStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.ballStartPos = [0,0,0.18]
        self.robotArmLength = 0.1
        self.inputVel = 0
        self.maxForce = 50

        planeId = p.loadURDF("simulator/custom_plane.urdf")
        self.boxId = p.loadURDF("simulator/10-balldroid.urdf",self.ballStartPos, self.ballStartOrientation)

        self.reset(0,[0,0,0])


    def __del__(self):
        p.disconnect()


    def reset(self, target, obstaclePos):
        
        p.resetSimulation()
        p.setGravity(0,0,-10)
        '''
        maptree = et.parse('simulator/custom_plane.urdf')
        maproot = maptree.getroot()
        maproot[2][2][0].set('xyz',str(obstaclePos[0]) + ' 0 0')
        maproot[2][3][0].set('xyz',str(obstaclePos[0]) + ' 0 0')
        maproot[4][2][0].set('xyz',str(obstaclePos[1]) + ' 0 0')
        maproot[4][3][0].set('xyz',str(obstaclePos[1]) + ' 0 0')
        maproot[6][2][0].set('xyz',str(obstaclePos[2]) + ' 0 0')
        maproot[6][3][0].set('xyz',str(obstaclePos[2]) + ' 0 0')
        maptree.write('simulator/custom_plane.urdf')
        '''
        planeId = p.loadURDF("simulator/custom_plane.urdf")
        self.boxId = p.loadURDF("simulator/10-balldroid.urdf",self.ballStartPos, self.ballStartOrientation)
        
        self.targetVel = target        
        self.inputVel = 0

        return np.concatenate([np.zeros(self.observation_space - 1),np.array([self.targetVel])])


    def step(self, action):
        self.inputVel = action

        # apply wheel velocity while step
        for i in range(self.step_size):
            p.setJointMotorControl2(self.boxId, 0, p.VELOCITY_CONTROL, targetVelocity = self.inputVel, force = self.maxForce)
            p.stepSimulation()

        # Get data
        ballPos, ballOrn = p.getBasePositionAndOrientation(self.boxId)
        ballVelLinear,ballVelAngular = p.getBaseVelocity(self.boxId)
        jointState = p.getJointState(self.boxId,0)
        robotPos,robotOri,_,_,_,_,_,robotAngVel = p.getLinkState(self.boxId,1,computeLinkVelocity = 1)
        robotOriEuler = p.getEulerFromQuaternion(robotOri)

        targetError = self.targetVel - ballVelLinear[0]

        # Make data array
        data = np.array([
            ballVelLinear[0],
            self.robotArmLength*robotOriEuler[1],
            self.robotArmLength*robotAngVel[1],
            self.robotArmLength*jointState[1],
            targetError
            ])
        
        # Estimate value
        fakeValue = -((self.targetVel-ballVelLinear[0])**2)

        done = False
        #time.sleep(1./240.)
        return data, fakeValue, done 
