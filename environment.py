import math
import random
import numpy as np
from core import World, Agent, Landmark
from copy import deepcopy
import pickle

locs = np.load("env/locs.npy")
vels = np.load("env/vels.npy")
hbs = np.load("env/hbs.npy")
distance_sensors = np.load("env/distance_sensors.npy")
range_agents = np.load("env/range_agents.npy")
#intervals = pickle.load(open("intervals.pkl","rb"))

class Scenario():
    def __init__(self):
        self.max_length = math.sqrt(161874)/2
        self.num_landmarks = 20#20
        self.num_agents = 2
        self.w0,self.w1,self.w2,self.w3 = 1,1,1,1
        self.T_m = 0.3
        self.lamda = 0.9
        self.margins = [4.24,36,0.424,1]#3*\sqrt(2)*\sigma

    def make_world(self, um):
        world = World(um=um)
        # add landmarks（cows）
        world.landmarks = [Landmark(i) for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
        # add agents
        world.agents = [Agent(i,len(world.landmarks)) for i in range(self.num_agents)]
        #world.agents[0].pos = np.array([-self.max_length/2,0])#np.zeros(world.dim_p)
        #world.agents[1].pos = np.array([self.max_length/2,0])#np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, initial_steps=0):
        world.set_ia()
        # random properties for agents
        world.reset()

    def reward(self, agent, world):
        diff_quality = np.zeros((4,),dtype=float)
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps,i],vels[world.steps,i],landmark.bl]# use actual value
            for idx in range(4):
                diff_quality[idx] += min(1,abs(agent.new_database[i][idx]-attributes[idx])/self.margins[idx])
        agent.pre_utility += -np.sum(diff_quality/(self.num_landmarks*4))-world.overload
        return agent.pre_utility

    def monitoring_quality(self, agent, world):
        # database: [sender ID,HR,AT,MXT,MNT,M,BL,T,D]
        diff_quality = np.zeros((4,),dtype=float)
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps,i],vels[world.steps,i],landmark.bl]# use actual value
            for idx in range(4):
                diff_quality[idx] += min(1,abs(agent.new_database[i][idx]-attributes[idx])/self.margins[idx])
        agent.mq = np.sum(diff_quality/(self.num_landmarks*4))#(self.lamda*agent.mq+diff_quality)/(1+self.lamda)
        return agent.mq
        #return self.mq/(self.num_landmarks*4)

    def vacuity(self, agent):
        return agent.mean_vacuity
    
    def dissonance(self, agent):
        return agent.mean_dissonance
    
    def freshness(self, agent):
        return agent.mean_freshness

    def energy_diff(self, agent):
        return -self.w3*math.pow(agent.mean_bl-self.T_m,2)

    def metrics(self, agent, world):
        mq = self.monitoring_quality(agent, world)
        vc = self.vacuity(agent)
        do = self.dissonance(agent)
        fs = self.freshness(agent)
        er = self.energy_diff(agent)
        ol = world.overload
        return mq, vc, do, fs, er, ol

class ENV_dqn(object):
    def __init__(self, length, scenario, nagents, um = False):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)#np.array([0,1,2,3],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world(um=um)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions,eval_set1 = set(),eval_set2 = set()):
        done = False
        temp_metric = []
        for i in range(self.nagents):
            self.state[i,self.steps] = actions[i] + 1
            self.world.agents[0].action = self.world.agents[1].action = actions[i]
        # advance world state
        #for i in range(self.nagents):
        self.world.agents[0].pre_utility = self.world.agents[1].pre_utility = 0
        while self.world.steps < (1+self.steps)*self.interval:
            self.world.step()
            self.world.step()
            temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(2)]
            temp_metric.append([self.sf.metrics(self.world.agents[i], self.world) for i in range(2)])
        self.steps += 1
        if self.steps == self.length:
            done = True
            while self.world.steps < 2878:
                self.world.step()
                self.world.step()
                temp_metric.append([self.sf.metrics(self.world.agents[i], self.world) for i in range(2)])
        #print("DRL:",self.world.steps)
        #print(self.area)
        #temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
        if done:
            #print(self.state)
            eval_set1.add(sum([self.state[0,i]*pow(3,i) for i in range(self.length)]))
            #eval_set2.add(sum([self.state[1,i]*pow(3,i) for i in range(self.length)]))
            r = sum(temp_r)
            #print(r)
        else:
            r = sum(temp_r)#r = [0 for i in range(self.nagents)]
        return self.state, r, done, eval_set1, eval_set2, temp_metric
    
class ENV_ppo(object):
    def __init__(self, length, scenario, nagents, um = False):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)#np.array([0,1,2,3],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world(um=um)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions,eval_set1 = set(),eval_set2 = set()):
        done = False
        temp_metric = []
        for i in range(self.nagents):
            self.state[i,self.steps] = actions[i] + 1
            self.world.agents[i].action = actions[i]
        # advance world state
        for i in range(self.nagents):
            self.world.agents[i].pre_utility = 0
        while self.world.steps < (1+self.steps)*self.interval:
            self.world.step()
            self.world.step()
            temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
            temp_metric.append([self.sf.metrics(self.world.agents[i], self.world) for i in range(2)])
        self.steps += 1
        if self.steps == self.length:
            done = True
            while self.world.steps < 2878:
                self.world.step()
                self.world.step()
                temp_metric.append([self.sf.metrics(self.world.agents[i], self.world) for i in range(2)])
        #print("DRL:",self.world.steps)
        #print(self.area)
        #temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
        if done:
            #print(self.state)
            eval_set1.add(sum([self.state[0,i]*pow(3,i) for i in range(self.length)]))
            eval_set2.add(sum([self.state[1,i]*pow(3,i) for i in range(self.length)]))
            r = temp_r
            #print(r)
        else:
            r = temp_r#r = [0 for i in range(self.nagents)]
        return self.state, r, done, eval_set1, eval_set2, temp_metric