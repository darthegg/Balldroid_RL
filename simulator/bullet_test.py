import pybullet as p
import time
import pybullet_data
import numpy
import xml.etree.ElementTree as et

obstaclePos1 = 1.55
obstaclePos2 = 2
obstaclePos3 = 3

maptree = et.parse('custom_plane.urdf')
maproot = maptree.getroot()

maproot[2][2][0].set('xyz',str(obstaclePos1) + ' 0 0')
maproot[2][3][0].set('xyz',str(obstaclePos1) + ' 0 0')
maproot[4][2][0].set('xyz',str(obstaclePos2) + ' 0 0')
maproot[4][3][0].set('xyz',str(obstaclePos2) + ' 0 0')
maproot[6][2][0].set('xyz',str(obstaclePos3) + ' 0 0')
maproot[6][3][0].set('xyz',str(obstaclePos3) + ' 0 0')


maptree.write('custom_plane.urdf')


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

p.setGravity(0,0,-10)
planeId = p.loadURDF("custom_plane.urdf")

ballStartPos = [0,0,0.175]
ballStartOrientation = p.getQuaternionFromEuler([0,0,0])

boxId = p.loadURDF("10-balldroid.urdf",ballStartPos, ballStartOrientation)

maxForce = 500
p.setJointMotorControl2(boxId,0,p.VELOCITY_CONTROL,targetVelocity = 4, force=maxForce)


startTime = time.time()

for i in range (10000):
    p.stepSimulation()

    if i%4 == 0:
        ballPos, ballOrn = p.getBasePositionAndOrientation(boxId)
        print("Ball_Pos_x : ",ballPos[0])

        ballVelLinear,ballVelAngular = p.getBaseVelocity(boxId)
        print("Ball Vel_x : ",ballVelLinear[0])

        jointState = p.getJointState(boxId,0)
        #print("Joint_Pos : ",jointState[0], "Joint_Vel : ",jointState[1])

        robotPos,robotOri,_,_,_,_,_,robotAngVel = p.getLinkState(boxId,1,computeLinkVelocity = 1)
        #print("Robot_Pos_x : ", robotPos[0],"Robot_Pos_z : ", robotPos[2])
        
        robotOriEuler = p.getEulerFromQuaternion(robotOri)
        #print("Robot_Ori_y : ", robotOriEuler[1])
        
        #print("Robot_AngVel_y : ", robotAngVel[1])

        data = [ballPos[0],ballPos[1],ballPos[2],ballVelLinear[0],ballVelLinear[1],ballVelLinear[2],robotOriEuler[0],robotOriEuler[1],robotOriEuler[2],robotAngVel[0],robotAngVel[1],robotAngVel[2]]

        if data[0] <= -2:
            done = True
            #reset
            p.resetSimulation()
            startTime = time.time()
            p.setGravity(0,0,-10)
            planeId = p.loadURDF("custom_plane.urdf")
            boxId = p.loadURDF("10-balldroid.urdf",ballStartPos, ballStartOrientation)
            p.setJointMotorControl2(boxId,0,p.VELOCITY_CONTROL,targetVelocity = 6, force=maxForce)

        else:
            done = False

        print(time.time() - startTime)

    time.sleep(1./240.)
            
p.disconnect()

