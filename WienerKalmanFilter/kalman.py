import numpy as np
import matplotlib.pyplot as plt
import random

class kalman():
    def __init__(self,state_transition_model,control_input_model,observation_model,process_covariance,
        observation_covariance,estimate_covariance,control_vector,state_vector):

        self.TM=state_transition_model
        self.CM=control_input_model
        self.OM=observation_model
        self.PC=process_covariance
        self.OC=observation_covariance
        self.EC=estimate_covariance
        self.CV=control_vector
        self.XV=state_vector
    def myPx(self):

        partc=np.dot(self.TM, self.EC)
        partd=np.dot(partc,self.TM.T)


        self.EC=partd+self.PC


        ######################################################
        parta=np.dot(self.TM, self.XV)
        partb=np.dot(self.CM, self.CV)

        self.XV=parta+partb
        ######################################
        return self.XV

    def myUP(self,obs):

        myp=np.dot(self.OM, self.XV)
        residual=obs-myp

        ########################################
        part1=np.dot(self.EC,self.OM.T)
        rc=np.dot(self.OM,part1)+self.OC
        rcinv=np.linalg.inv(rc)
        ##################################################
        part2=np.dot(self.EC, self.OM.T)
        gain=np.dot(part2,rcinv)

        self.XV=self.XV+np.dot(gain,residual)
        ###############################################
        I=np.eye(self.TM.shape[0])

        part3=np.dot(gain,self.OM)
        part4=I-part3
        part5=np.dot(part4,self.EC)
        part6=np.dot(part5,part4.T)

        #################################
        part7=np.dot(gain,self.OC)
        part8=np.dot(part7,gain.T)
        #######################################
        self.EC=part6+part8
        

def myfunc(x):
    d=2*pow(x,3)+3*pow(x,2)+2*x-1+random.randint(-5,5)

    return d

###################################################################
state_transition_model=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
control_input_model=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
observation_model=np.array([0.1, 0.3, 0.4]).reshape(1, 3)
process_covariance=np.array([[0.3, 0.2, 0.0], [0.1, 0.02, 0.0], [0.4, 0.1, 0.2]])
observation_covariance=np.array([0.5]).reshape(1, 1)
estimate_covariance=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
control_vector=np.array([1, 0, 0]).reshape(3, 1)
state_vector=np.array([1, 0, 0]).reshape(3, 1)
##################################################################
kf=kalman(state_transition_model=state_transition_model,control_input_model=control_input_model,observation_model=observation_model,process_covariance=process_covariance,
        observation_covariance=observation_covariance,estimate_covariance=estimate_covariance,control_vector=control_vector,state_vector=state_vector)
###################################################################################
xs=np.linspace(-5, 5, 100)

data=[]
ps=[]

###################################
for i in range(len(xs)):
    dd=myfunc(xs[i])
    data.append(dd)
    ###########################################3
    see=kf.myPx()

    res=(np.dot(observation_model,see)[0])[0]
    #print(res)
    ps.append(res)
    kf.myUP(dd)
#########################################

plt.plot(range(len(data)),data,label='real data')
plt.plot(range(len(ps)),ps,label='prediction')
plt.legend()
plt.show()
