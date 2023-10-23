

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
import gym
from gym import spaces
import math
import matplotlib.pyplot as plt
import os


MAX_EPISODES = 50
MAX_EP_STEPS = 75
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)][0]            
# you can try different target replacement strategies
MEMORY_CAPACITY = 1000   #100    #10000
BATCH_SIZE = 32
np.random.seed(1)
tf.set_random_seed(1)
'''
data1 = np.linspace(0,1,20)
data2 = np.linspace(0,1,20)
data = np.vstack((data1,data2)).T
data = np.expand_dims(data,0).repeat(5,axis=0)
'''
data = np.load(r'DJI_01870.npy')
data[:,1:,0] *= ((data[:,1:,0]>90) *1 + (data[:,1:,0]<90) *0.2 )

OUTPUT_GRAPH = False

# In[]
class env:   
    def __init__(self,data):
        self.data = data
        self.max_speed=0.1  #最大速度
        self.action_space = spaces.Box(low=-self.max_speed, \
                        high = self.max_speed ,shape=(2,),dtype = np.float32)
       
        self.inform_range = 0.3 #感知范围
        self.observation_space = spaces.Box(low=-self.inform_range , high=self.inform_range ,shape=(6,) ,dtype=np.float32)
        # 轨迹  x，y轴上序列  读取10秒序列
        self.observation_tj = spaces.Box(low=0 , high=1 ,shape=(20,) ,dtype=np.float32)
    def inform(self, j):    #得到所有人在某时刻j感知到的环境
        s = np.empty(shape = [0,6])  #将所有人感知到的环境进行汇总
        for i in range(self.data.shape[0]):  #循环第一个维度，得到每个人i在某时刻j感知到的环境
            s_one = np.empty(shape = [0,2])   #每个人检测到的环境 感知范围内每个存在的人距本人的距离
            for k in range(self.data.shape[0]): #循环第一个维度（某个人）
                if k != i: #排除感知自己
                    delta_x = self.data[k,j,0]-self.data[i,j,0]
                    delta_y = self.data[k,j,1]-self.data[i,j,1]
                    d = math.sqrt(delta_x**2 + delta_y**2)   #与某个人之间的距离
                    if d <= self.inform_range:
                        s_one = np.append(s_one,[[delta_x,delta_y]],axis=0)
            s2 = s_one**2
            d2 = s2.sum(axis=1)   #搜索出的每个人与其的距离
            index = np.argsort(d2)   #距离由近到远的序号排列
            s_one = s_one.take(index,axis=0)  #将由delta_x delta_y构成的数组按序号进行排序
            if s_one.shape[0]>3:  #仅感知最近的三个人
                s_one = s_one[:3]
            if s_one.shape[0]<3:  #为控制shape为(3,2)
                lack = 3 - s_one.shape[0]   #缺少的人数
                for i in range(lack):
                    s_one = np.append(s_one,[[0,0]],axis=0)  #缺少的以0填满
            s_one = s_one.reshape(6)   #将s_one改为6维向量
            s = np.append(s,[s_one],axis=0)  #汇总 #(None,6)
            
        return s  #以delta_x,delta_y构成的数组，且感知人数为3人，shape为（None,6）
        
    # def inform_2(self, s_sum):    #得到所有人在汇总的场景s_sum（坐标）下感知到的环境
    #     s = np.empty(shape = [0,6])  #将所有人感知到的环境进行汇总
    #     for i in range(s_sum.shape[0]):  #循环第一个维度，得到每个人i在某时刻j感知到的环境
    #         s_one = np.empty(shape = [0,2])   #每个人检测到的环境 感知范围内每个存在的人距本人的距离
    #         for k in range(s_sum.shape[0]): #循环第一个维度（某个人）
    #             if k != i: #排除感知自己
    #                 delta_x = s_sum[k,0]-s_sum[i,0]
    #                 delta_y = s_sum[k,1]-s_sum[i,1]
    #                 d = math.sqrt(delta_x**2 + delta_y**2)   #与某个人之间的距离
    #                 if d <= self.inform_range:
    #                     s_one = np.append(s_one,[[delta_x,delta_y]],axis=0)
    #         s2 = s_one**2
    #         d2 = s2.sum(axis=1)   #搜索出的每个人与其的距离
    #         index = np.argsort(d2)   #距离由近到远的序号排列
    #         s_one = s_one.take(index,axis=0)  #将由delta_x delta_y构成的数组按序号进行排序
    #         if s_one.shape[0]>3:  #仅感知最近的三个人
    #             s_one = s_one[:3]
    #         if s_one.shape[0]<3:  #为控制shape为(3,2)
    #             lack = 3 - s_one.shape[0]   #缺少的人数
    #             for i in range(lack):
    #                 s_one = np.append(s_one,[[0,0]],axis=0)  #缺少的以0填满
    #         s_one = s_one.reshape(6)   #将s_one改为6维向量
    #         s = np.append(s,[s_one],axis=0)  #汇总 #(None,6)
            
        # return s  #以delta_x,delta_y构成的数组，且感知人数为3人，shape为（None,6）
    def tj_past(self,j):
        tj = np.empty(shape = [0,20])
        for i in range(self.data.shape[0]):  #循环第一个维度，得到每个人i在某时刻j时的前10个时间单位的轨迹
            tj_one = data[i,j-9:j+1] #包括当前所在位置
            tj_one = tj_one.reshape(20)
            tj = np.append(tj,[tj_one],axis = 0) #汇总
        return tj  # (None,10✖2)
    
    def step(self, a ,j, s): #j时刻 当前的s场景
        '''输入每个人的动作数组，更新下一秒整体的环境
        输出1：（借助inform函数，输出每人感知到的环境组成的数组）
        输出2：输出整体的回报 -mse
        '''
        s_sum = s[:,25:] + a
        diff = self.data[:,j+1] - s_sum #计算每个人的x，y坐标差值
        #diff_2 = diff ** 2    #平方
        diff_abs = abs(diff[:,0]) + abs(diff[:,1])#绝对值
        r = np.empty(shape = [0])
        for i in range(self.data.shape[0]):
            #mse = diff_2[i].mean() #均值 得到每个人的mse
            #r = np.append(r,[-mse],axis = 0)
            mae = diff_abs[i]
            r = np.append(r,[mae],axis = 0)
        #训练时使r增大       
        
        
        
        s_next = self.inform(j+1) # 直接感知下一秒所有人实际所在的位置，实际上为一种单步训练，这是为了防止第一步偏离后，在后续步中，学习到不应该有的动作
        tj = self.tj_past(j+1)  #直接得到下一秒所有人之前的轨迹
        s_next = np.concatenate((s_next,tj),axis = 1)  #(none,26)
        return s_next, r, s_sum #实际中下一秒的场景以及轨迹（每个人感知后的） -mse 预测动作下得到的全体的下一秒场景
        #return s_next,tj_next实际,r,s_sum
    
    def step2(self,a,j,s):   #不断更新虚拟的场景
        s_sum = s[:,24:] + a
        diff = self.data[:,j+1] - s_sum #计算每个人的x，y坐标差值
        #diff_2 = diff ** 2    #平方
        diff_abs = abs(diff[:,0]) + abs(diff[:,1])#绝对值
        r = np.empty(shape = [0])
        for i in range(self.data.shape[0]):
            #mse = diff_2[i].mean() #均值 得到每个人的mse
            #r = np.append(r,[-mse],axis = 0)
            mae = diff_abs[i]
            r = np.append(r,[-mae],axis = 0)
        #训练时使r增大 
        s_sim_next = np.empty(shape = [0,26])
        for kk in range(self.data.shape[0]): #。再把action和s_sum拆开 每个人kk
            s_one = np.empty(shape = [0,2])   #每个人检测到的环境 感知范围内每个存在的人距本人的距离
            for k in range(s_sum.shape[0]): #循环第一个维度（某个人） s_sum.shape:(None,2)
                if k != kk: #排除感知自己
                    delta_x = s_sum[k,0]-s_sum[kk,0]
                    delta_y = s_sum[k,1]-s_sum[kk,1]
                    d = math.sqrt(delta_x**2 + delta_y**2)   #与某个人之间的距离
                    if d <= 0.3: #self.inform_range:
                        s_one = np.append(s_one,[[delta_x,delta_y]],axis=0)
            s2 = s_one**2
            d2 = s2.sum(axis=1)   #搜索出的每个人与其的距离
            index = np.argsort(d2)   #距离由近到远的序号排列
            s_one = s_one.take(index,axis=0)  #将由delta_x delta_y构成的数组按序号进行排序
            if s_one.shape[0]>3:  #仅感知最近的三个人
                s_one = s_one[:3]
            if s_one.shape[0]<3:  #为控制shape为(3,2)
                lack = 3 - s_one.shape[0]   #缺少的人数
                for _ in range(lack):
                    s_one = np.append(s_one,[[0,0]],axis=0)  #缺少的以0填满
            s_one = s_one.reshape(6)  #转化为6维向量
            #轨迹
            tj_one = s[kk,6:] #包括当前所在位置
            tj_one = np.concatenate((tj_one[2:],s_sum[kk]),axis = 0) #!!!!!!
            
            tj_one = tj_one.reshape(20)
            s_one = np.concatenate((s_one,tj_one),axis = 0)
            s_sim_next = np.append(s_sim_next,[s_one],axis=0)
            # print("action",action)
            # print("s",s[:,30:])
            # print("s_pre_next",s_pre_next[:,30:])
        return s_sim_next,r
    
    def step_k(self,a,k,j,s):   #单独更新人的环境
        xy_k_next = s[24:] + a
        diff = self.data[k,j+1] - xy_k_next #计算每个人的x，y坐标差值
         
        diff_abs = abs(diff[0]) + abs(diff[1])#绝对值
        r = -diff_abs  #mae
        
        s_one = np.empty(shape = [0,2])    
        for kk in range(self.data.shape[0]): #循环第一个维度（某个人） kk检测到的人 k本人
            if kk != k: #排除感知自己
                delta_x = self.data[kk,j,0]-xy_k_next[0]
                delta_y = self.data[kk,j,1]-xy_k_next[1]
                d = math.sqrt(delta_x**2 + delta_y**2)   #与某个人之间的距离
                if d <= self.inform_range:
                    s_one = np.append(s_one,[[delta_x,delta_y]],axis=0)
        s2 = s_one**2
        d2 = s2.sum(axis=1)   #搜索出的每个人与其的距离
        index = np.argsort(d2)   #距离由近到远的序号排列
        s_one = s_one.take(index,axis=0)  #将由delta_x delta_y构成的数组按序号进行排序
        if s_one.shape[0]>3:  #仅感知最近的三个人
            s_one = s_one[:3]
        if s_one.shape[0]<3:  #为控制shape为(3,2)
            lack = 3 - s_one.shape[0]   #缺少的人数
            for i in range(lack):
                s_one = np.append(s_one,[[0,0]],axis=0)  #缺少的以0填满
        s_one = s_one.reshape(6)   #将s_one改为6维向量
        tj_one = s[6:] #包括当前所在位置
        tj_one = np.concatenate((tj_one[2:],xy_k_next),axis = 0) #!!!!!!
        
        tj_one = tj_one.reshape(20)
        s_one = np.concatenate((s_one,tj_one),axis = 0)
        
        # print("action",action)
        # print("s",s[:,30:])
        # print("s_pre_next",s_pre_next[:,30:])
        return s_one,r
        
     
# action = np.array([[0.01, 0.02],[0.1, 0.2],[0.1, 0.2],[0.1, 0.2]])
# env1 = env(data)
# s = env1.inform(9)
# tj = env1.tj_past(9)
# s = np.concatenate((s,tj),axis = 1)
# s_sim_next,r= env1.step2(action,9,s)      #下一秒的场景，以及得到的r
# s_sim_next2,r= env1.step2(action,10,s_sim_next)


# In[]网络
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):  #Actor 范围内
            # input s, output a
            # 这个网络用于及时更新参数
            self.a = self._build_net(S, scope='eval_net', trainable=True)  #调用自建的_build_net函数 

            # input s_, output a, get a_ for critic
            # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)
            
        #得到eval和target网络的参数
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':  #完全替换参数
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:      #部分替换参数
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            s_inf, s_tj = tf.split(s, [6, 20], axis = 1)  # 0是None
            lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(32,forget_bias=1)
            s_tj = tf.reshape(s_tj,shape = (-1,10,2))
            outputs_tj0,_=tf.nn.dynamic_rnn(lstm_layer,s_tj,dtype=tf.float32)
            outputs_tj = outputs_tj0[:,-1,:]
            outputs_inf = tf.layers.dense(s_inf, 32, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            #net = outputs_tj
            net = tf.concat([outputs_inf,outputs_tj],1)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1   #

    def choose_action(self, s):  #写s时不需再括号，直接[1,2,3],不用[[1,2,3]]
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):  #PG
            # ys = policy; 策略
            # xs = policy's parameters; 策略网络参数
            # a_grads = the gradients of the policy to get more Q 这个策略获得更多q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            #Adam优化器
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy 
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)#可训练

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)#不可训练    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
            
        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_


        with tf.variable_scope('TD_error'):
            #loss   
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))  
            print(self.loss)
        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)
        
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            
            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
            #print(self.a_grads)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'  #<时发生
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
    

# In[] 运行部分
env1 = env(data)

state_dim = env1.observation_space.shape[0] + env1.observation_tj.shape[0]
action_dim = env1.action_space.shape[0]
action_bound = env1.action_space.high  #high是一个1维的队列


# 设置神经网络的参数（状态 汇报 下一刻状态）
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
# with tf.name_scope('TJ'):  #x,y轴轨迹
#     TJX = tf.placeholder(tf.float32, shape=[None, trajectory_dim,2], name='tjx')
# with tf.name_scope('TJ_'): #x,y轴轨迹（下一刻）
#     TJX_ = tf.placeholder(tf.float32, shape=[None, trajectory_dim,2], name='tjx_')


sess = tf.Session()

# 创建actor和critic神经网络（两者相互联系）
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)


if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)
    
var = 0.005  # 动作值的方差  （mean-36，mean + 36）
#实际的运行部分
t1 = time.time()



for i in range(MAX_EPISODES): 
    env1 = env(data)  #每个场景下的环境
    s = env1.inform(10)
    tj = env1.tj_past(10)
    s = np.concatenate((s,tj),axis = 1)
    
    ep_reward = 0
    MAX_EP_STEPS = data.shape[1]-2

    
    
    
    for k in range(data.shape[0]):
        ep_reward = 0
        env1 = env(data)  #每个场景下的环境
        s = env1.inform(10)
        tj = env1.tj_past(10)
        s = np.concatenate((s,tj),axis = 1)[k]
        
        for j in range(10,MAX_EP_STEPS-1):
            if (data[k,j-9:j+2]<90).all():
                a = actor.choose_action(s)
                            #为行动加入随机
                a = np.clip(np.random.normal(a, var), -0.1, 0.1)
                #print(a)
            else:
                a = data[k,j+1] - s[24:]
            s_k_next,r = env1.step_k(a,k,j,s)
            
            if (s<90).all() and (s_k_next<90).all():
                M.store_transition(s, a, r, s_k_next)
                

            

            if M.pointer > MEMORY_CAPACITY:
                #var *= .9995    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]
                #print("b_s",b_s[0],"b_a", b_a[0],"b_r", b_r[0],"b_s_", b_s_[0])
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_k_next
            ep_reward += r

            if j == MAX_EP_STEPS-2:
                print('Episode:', i,'Number:',k, ' Reward: %f' % ep_reward, 'Explore: %f' % var, )
                

print('Running time: ', time.time()-t1)
'''
    啧，s应该从现在的（3，2）改为（6，）
'''

# In[]仿真

plt.ion()
plt.figure(num=2, figsize=(3, 6))
# plt.xlim((0, 1))
# plt.ylim((0, 1))
c = ['r','g','b','y','c','m','k','w']

for i in range(1,10): 
    plt.xlim((-1, 2))
    plt.ylim((-1, 2))
    for k in range(data.shape[0]):   
        if data[k,i,0]<90 and data[k,i+1,0]<90:
            plt.plot([data[k,i,0],data[k,i+1,0]],[data[k,i,1],data[k,i+1,1]],color = c[k])
    scatter = plt.scatter(data[:,i+1,0],data[:,i+1,1],color = 'r')
    
    plt.pause(0.1)

    scatter.remove()  #清屏


env1 = env(data)  #每个场景下的环境
s = env1.inform(10)
tj = env1.tj_past(10)
s = np.concatenate((s,tj),axis = 1)

for j in range(10,MAX_EP_STEPS-1):  #每个场景下的秒数（步数）
    action = np.empty(shape = [0,2])     #汇总后的action
    plt.xlim((-0.5, 1.5))
    plt.ylim((-0.5, 1.5))    
    
    for k in range(data.shape[0]): #每个场景下的每个人
        
        if (data[k,j-9:j+2]<90).all():
            a = actor.choose_action(s[k])
            #为行动加入随机
            a = np.clip(np.random.normal(a, var), -0.1, 0.1)
        else:
            a = data[k,j+1] - s[k][24:]
        if s[k][24]<90 and (s[k][24]+a[0])<90 :
            plt.plot([s[k][24],s[k][24]+a[0]],[s[k][25],s[k][25]+a[1]],color = c[k])
        
        
        action = np.append(action,[a],axis=0)
    scatter = plt.scatter([s[:,24]+action[:,0]],[s[:,25]+action[:,1]],color = 'r')

    s_sim_next, r  = env1.step2(action,j,s)  


    print("action",action)
    print("s",s[:,24:])
    print("s_sim_next",s_sim_next[:,24:])
    plt.pause(0.1)
    scatter.remove()  #清屏
    s = s_sim_next
    
                                           
plt.ioff()                                           
plt.show()

