import gymnasium as gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from .agent import BaseAgent
from .common import utils

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments
    简单的存储n步的经验采集样本，用于单个或者多个环境中

    Every experience contains n list of Experience entries
    每个经验样本集都包含n步的经验
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used 环境信息或者环境信息列表
        :param agent: callable to convert batch of states into actions to take 代理信息
        :param steps_count: count of steps to track for every experience chain 需要追溯多少步以前的记录
        :param steps_delta: how many steps to do between experience items todo
        :param vectorized: support of vectorized envs from OpenAI universe 
        """
        # 判断经验传入的参数类型是否正确
        # 并存储到成员变量
        assert isinstance(env, (gym.Env, gym.vector.VectorEnv, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        # self.pool: 存储游戏环境
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        # 这个接口就是运行环境并获取观测值，填充经验缓冲区的地方
        # 调用这个接口后，每次遍历都会使用神经网络预测当前状态下的
        # 执行动作，使用yield实现
        # 如果游戏一旦结束，将会把最后一次的游戏状态存储到total_rewards（包括执行的动作，奖励值，游戏的环境状态）
        # 
        # return: 
        # 
        
        # states: 存储每一次的环境观测值
        # agent_states: 存储游戏网络的代理的初始状态
        # histories: 存储的好像是一个队列，长度为step_ount，存储的应该是历史记录，存储多少步以前的状态等信息 todo
        # cur_rewards: 存储每轮游戏观测的激励，游戏初始化时存储的是0.0
        # cur_steps: 存储当前观测结果的步数（todo），游戏初始化时存储的是0
        # agent_states： 代理状态, 作用1：表示当前游戏主体所处状态的评价值（Q值） todo 作用
        # 看代码后，发现这个agnet states的状态只有在ExperienceSourceRollouts中才有实际的作用

        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        # 存储每次环境观测值的长度（因为矢量的环境其返回的结果值是一个不定长度的列表）
        # 每个索引对应着states中对应索引的长度
        env_lens = []
        # 遍历每一个游戏环境
        # 这一大段的循环作用是初始化环境，得到初始化结果
        for env in self.pool:
            # 每一次遍历时，重置游戏环境
            obs, obs_info = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # 如果支持矢量计算，那么可以将多个环境的输出结果，拼接到矩阵向量中，直接进行计算，效率比单个计算高
            if self.vectorized:
                # 矢量环境
                # 获取单词观察结果的向量长度
                obs_len = len(obs)
                # 将当前状态结果列表（应该是包含了环境状态，激励，动作，是否结束等信息）存储在states
                states.extend(obs)
            else:
                # 非矢量环境下，其观测的结果就是一个标量，简单说就是一个动作值，所以
                # 长度是1
                obs_len = 1
                # 将结果存储在status中
                states.append(obs)
            env_lens.append(obs_len)
            
            # 遍历本次环境观测的结果
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count)) # 创建对应环境观测结果历史缓存队列
                cur_rewards.append(0.0)  # 存储最初是状态下的激励，为0
                cur_steps.append(0) # 存储当前行走的部署，为0
                agent_states.append(self.agent.initial_state()) # 存储代理状态，reset环境时，代理状态是初始状态
        
        # 遍历索引
        # 从里这开始，应该就是尝试运行游戏了
        iter_idx = 0
        while True: 
            actions = [None] * len(states) # todo
            states_input = []
            states_indices = []
            # 遍历每一次的存储的观测状态
            # 对于非矢量环境来说，idx仅仅对应一个当前状态，但是对于矢量环境来说，idx对应当前获取的每个一观测值的索引
            # 这一个大循环的作用应该是，根据环境执行得到执行的动作结果
            # todo 有一个问题，矢量环境获取的多个观测结果这里怎么分辨
            for idx, state in enumerate(states):
                if state is None:
                    # 如果状态是空的，则使用环境进行随机选择一个动作执行， 另外这里假设的所有的环境都有相同的动作空间，
                    # 所以这里仅使用0索引环境进行随机动作采样
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    # 如果状态非空，则将当前的存储在states环境观测值存储在states_input中
                    # 并存储当前的索引
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                # 如果观测的状态列表非空，则将状态输入的神经网络环境代理中，获取将要执行的动作
                # 而agent_staes根据源码，发现并未做处理
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                # 遍历每一个状态所要执行的动作
                for idx, action in enumerate(states_actions):
                    # 获取当前动作对应的状态的索引位置，有上面106行的代码可知
                    g_idx = states_indices[idx]
                    # 将执行的动作存储在与状态相对应的索引上
                    actions[g_idx] = action
                    # 代理状态（todo 作用）
                    agent_states[g_idx] = new_agent_states[idx]
            # 将动作按照原先存储的每个环境得到的观测结果长度，按照环境数组进行重新分割分组
            grouped_actions = _group_list(actions, env_lens)
            
            # 因为存在一个大循环，存储每个环境的起始索引位置
            global_ofs = 0
            # 遍历每一个环境
            # 这一个大循环是将上一个循环中得到的执行动作应用到实际的环境中
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    # 这里action_n是一个list，也就是说矢量环境的输入的一个多维
                    # 如果是矢量的环境，则直接执行动作获取下一个状态，激励，是否结束等观测值
                    next_state_n, r_n, is_done_n, truncated, _ = env.step(action_n)
                    is_done_n = np.logical_or(is_done_n, truncated)
                else:
                    # 如果不是矢量环境，则需要将动作的第一个动作发送到env中获取相应的观测值（这里之所以是[0]，因为为了和矢量环境统一，即时是一个动作也会以列表的方式存储）
                    next_state, r, is_done, truncated, _ = env.step(action_n[0])
                    is_done = is_done or truncated
                    # 这个操作是为了和矢量环境统一
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]
                
                # 遍历每一次的动作所得到的下一个状态、激励、是否结束
                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # 获取当前缓存的索引位置
                    idx = global_ofs + ofs
                    # 获取初始环境的状态
                    # 因为action_n存储的就是每一个状态下所执行的动作，所以这里直接使用idx提取对应的状态
                    state = states[idx]
                    # 获取一个历史队列，此时队列为空
                    history = histories[idx]
                    
                    # 这里利用的idx来区分每一个状态执行的动作所对应的激励值
                    # 将获取的激励值存储在缓存中
                    cur_rewards[idx] += r
                    # 将当前状态以及执行的动作，执行的步骤次数存在起来
                    cur_steps[idx] += 1
                    # 如果状态非空，则（当前状态，所执行的动作对应的历史缓存队列）将当前状态存储在history中
                    # 所以一个样本可能就是这样对应一个队列数据
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    # 如果达到了采集的步数并且遍历索引达到了两个经验样本的指定差值，则将样本返回，待外界下一次继续获取时，从这里继续执行
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    # 更新states，表示当前动作执行后状态的改变
                    # 将动作设置为动作执行后的下一个状态，因为idx表示当前运行的环境状态的变更
                    states[idx] = next_state
                    if is_done:
                        # 如果游戏结束，如果存储的历史数据小于指定的长度，则直接返回
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        # 弹出最左侧的历史数据，返回给外部获取数据
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        # 将当前状态+动作执行后得到的激励存储在total_rewards队列中
                        self.total_rewards.append(cur_rewards[idx])
                        # 这个当前状态+动作执行的次数也存储起来
                        self.total_steps.append(cur_steps[idx])
                        # 重置状态
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset()[0] if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                        
                # 将起始索引设置为下一个环境的起始位置
                global_ofs += len(action_n)
            # 遍历索引+1
            iter_idx += 1

    def pop_total_rewards(self):
        """
        返回所有采集的样本，并清空缓存
        """
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class ExperienceSourceRAW:
    """
    Simple n-step experience source using single or multiple environments
    简单的存储n步的经验采集样本，用于单个或者多个环境中

    Every experience contains n list of Experience entries
    每个经验样本集都包含n步的经验
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used 环境信息或者环境信息列表
        :param agent: callable to convert batch of states into actions to take 代理信息
        :param steps_count: count of steps to track for every experience chain 需要追溯多少步以前的记录
        :param steps_delta: how many steps to do between experience items todo
        :param vectorized: support of vectorized envs from OpenAI universe 
        """
        # 判断经验传入的参数类型是否正确
        # 并存储到成员变量
        assert isinstance(env, (gym.Env, gym.vector.VectorEnv, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        # self.pool: 存储游戏环境
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        # 这个接口就是运行环境并获取观测值，填充经验缓冲区的地方
        # 调用这个接口后，每次遍历都会使用神经网络预测当前状态下的
        # 执行动作，使用yield实现
        # 如果游戏一旦结束，将会把最后一次的游戏状态存储到total_rewards（包括执行的动作，奖励值，游戏的环境状态）
        # 
        # return: 
        # 
        
        # states: 存储每一次的环境观测值
        # agent_states: 存储游戏网络的代理的初始状态
        # histories: 存储的好像是一个队列，长度为step_ount，存储的应该是历史记录，存储多少步以前的状态等信息 todo
        # cur_rewards: 存储每轮游戏观测的激励，游戏初始化时存储的是0.0
        # cur_steps: 存储当前观测结果的步数（todo），游戏初始化时存储的是0
        # agent_states： 代理状态, 作用1：表示当前游戏主体所处状态的评价值（Q值） todo 作用
        # 看代码后，发现这个agnet states的状态只有在ExperienceSourceRollouts中才有实际的作用

        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        # 存储每次环境观测值的长度（因为矢量的环境其返回的结果值是一个不定长度的列表）
        # 每个索引对应着states中对应索引的长度
        env_lens = []
        # 遍历每一个游戏环境
        # 这一大段的循环作用是初始化环境，得到初始化结果
        for env in self.pool:
            # 每一次遍历时，重置游戏环境
            obs, obs_info = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # 如果支持矢量计算，那么可以将多个环境的输出结果，拼接到矩阵向量中，直接进行计算，效率比单个计算高
            if self.vectorized:
                # 矢量环境
                # 获取单词观察结果的向量长度
                obs_len = len(obs)
                # 将当前状态结果列表（应该是包含了环境状态，激励，动作，是否结束等信息）存储在states
                states.extend(obs)
            else:
                # 非矢量环境下，其观测的结果就是一个标量，简单说就是一个动作值，所以
                # 长度是1
                obs_len = 1
                # 将结果存储在status中
                states.append(obs)
            env_lens.append(obs_len)
            
            # 遍历本次环境观测的结果
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count)) # 创建对应环境观测结果历史缓存队列
                cur_rewards.append(0.0)  # 存储最初是状态下的激励，为0
                cur_steps.append(0) # 存储当前行走的部署，为0
                agent_states.append(self.agent.initial_state()) # 存储代理状态，reset环境时，代理状态是初始状态
        
        # 遍历索引
        # 从里这开始，应该就是尝试运行游戏了
        iter_idx = 0
        while True: 
            actions = [None] * len(states) # todo
            states_input = []
            states_indices = []
            # 遍历每一次的存储的观测状态
            # 对于非矢量环境来说，idx仅仅对应一个当前状态，但是对于矢量环境来说，idx对应当前获取的每个一观测值的索引
            # 这一个大循环的作用应该是，根据环境执行得到执行的动作结果
            # todo 有一个问题，矢量环境获取的多个观测结果这里怎么分辨
            for idx, state in enumerate(states):
                if state is None:
                    # 如果状态是空的，则使用环境进行随机选择一个动作执行， 另外这里假设的所有的环境都有相同的动作空间，
                    # 所以这里仅使用0索引环境进行随机动作采样
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    # 如果状态非空，则将当前的存储在states环境观测值存储在states_input中
                    # 并存储当前的索引
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                # 如果观测的状态列表非空，则将状态输入的神经网络环境代理中，获取将要执行的动作
                # 而agent_staes根据源码，发现并未做处理
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                # 遍历每一个状态所要执行的动作
                for idx, action in enumerate(states_actions):
                    # 获取当前动作对应的状态的索引位置，有上面106行的代码可知
                    g_idx = states_indices[idx]
                    # 将执行的动作存储在与状态相对应的索引上
                    actions[g_idx] = action
                    # 代理状态（todo 作用）
                    agent_states[g_idx] = new_agent_states[idx]
            # 将动作按照原先存储的每个环境得到的观测结果长度，按照环境数组进行重新分割分组
            grouped_actions = _group_list(actions, env_lens)
            
            # 因为存在一个大循环，存储每个环境的起始索引位置
            global_ofs = 0
            # 遍历每一个环境
            # 这一个大循环是将上一个循环中得到的执行动作应用到实际的环境中
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    # 这里action_n是一个list，也就是说矢量环境的输入的一个多维
                    # 如果是矢量的环境，则直接执行动作获取下一个状态，激励，是否结束等观测值
                    next_state_n, r_n, is_done_n, truncated, _ = env.step(action_n)
                    is_done_n = np.logical_or(is_done_n, truncated)
                else:
                    # 如果不是矢量环境，则需要将动作的第一个动作发送到env中获取相应的观测值（这里之所以是[0]，因为为了和矢量环境统一，即时是一个动作也会以列表的方式存储）
                    next_state, r, is_done, truncated, _ = env.step(action_n[0])
                    is_done = is_done or truncated
                    # 这个操作是为了和矢量环境统一
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]
                
                # 遍历每一次的动作所得到的下一个状态、激励、是否结束
                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # 获取当前缓存的索引位置
                    idx = global_ofs + ofs
                    # 获取初始环境的状态
                    # 因为action_n存储的就是每一个状态下所执行的动作，所以这里直接使用idx提取对应的状态
                    state = states[idx]
                    # 获取一个历史队列，此时队列为空
                    history = histories[idx]
                    agent_state = agent_states[idx]
                    
                    # 这里利用的idx来区分每一个状态执行的动作所对应的激励值
                    # 将获取的激励值存储在缓存中
                    cur_rewards[idx] += r
                    # 将当前状态以及执行的动作，执行的步骤次数存在起来
                    cur_steps[idx] += 1
                    # 如果状态非空，则（当前状态，所执行的动作对应的历史缓存队列）将当前状态存储在history中
                    # 所以一个样本可能就是这样对应一个队列数据
                    if state is not None:
                        history.append((state, action, r, is_done, next_state, agent_state, env_idx))
                    # 如果达到了采集的步数并且遍历索引达到了两个经验样本的指定差值，则将样本返回，待外界下一次继续获取时，从这里继续执行
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    # 更新states，表示当前动作执行后状态的改变
                    # 将动作设置为动作执行后的下一个状态，因为idx表示当前运行的环境状态的变更
                    states[idx] = next_state
                    if is_done:
                        # 如果游戏结束，如果存储的历史数据小于指定的长度，则直接返回
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        # 弹出最左侧的历史数据，返回给外部获取数据
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        # 将当前状态+动作执行后得到的激励存储在total_rewards队列中
                        self.total_rewards.append(cur_rewards[idx])
                        # 这个当前状态+动作执行的次数也存储起来
                        self.total_steps.append(cur_steps[idx])
                        # 重置状态
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset()[0] if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                        
                # 将起始索引设置为下一个环境的起始位置
                global_ofs += len(action_n)
            # 遍历索引+1
            iter_idx += 1

    def pop_total_rewards(self):
        """
        返回所有采集的样本，并清空缓存
        """
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
    


class BaseAgentState:

    def update(self, obs, action, reward, done, next_obs):
        raise NotImplementedError


class ExperienceSourceRecurrent:
    """
    Simple n-step experience source using single or multiple environments
    简单的存储n步的经验采集样本，用于单个或者多个环境中

    Every experience contains n list of Experience entries
    每个经验样本集都包含n步的经验
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used 环境信息或者环境信息列表
        :param agent: callable to convert batch of states into actions to take 代理信息
        :param steps_count: count of steps to track for every experience chain 需要追溯多少步以前的记录
        :param steps_delta: how many steps to do between experience items todo
        :param vectorized: support of vectorized envs from OpenAI universe 
        """
        # 判断经验传入的参数类型是否正确
        # 并存储到成员变量
        assert isinstance(env, (gym.Env, gym.vector.VectorEnv, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        # self.pool: 存储游戏环境
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        # 这个接口就是运行环境并获取观测值，填充经验缓冲区的地方
        # 调用这个接口后，每次遍历都会使用神经网络预测当前状态下的
        # 执行动作，使用yield实现
        # 如果游戏一旦结束，将会把最后一次的游戏状态存储到total_rewards（包括执行的动作，奖励值，游戏的环境状态）
        # 
        # return: 
        # 
        
        # states: 存储每一次的环境观测值
        # agent_states: 存储游戏网络的代理的初始状态
        # histories: 存储的好像是一个队列，长度为step_ount，存储的应该是历史记录，存储多少步以前的状态等信息 todo
        # cur_rewards: 存储每轮游戏观测的激励，游戏初始化时存储的是0.0
        # cur_steps: 存储当前观测结果的步数（todo），游戏初始化时存储的是0
        # agent_states： 代理状态, 作用1：表示当前游戏主体所处状态的评价值（Q值） todo 作用
        # 看代码后，发现这个agnet states的状态只有在ExperienceSourceRollouts中才有实际的作用

        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        # 存储每次环境观测值的长度（因为矢量的环境其返回的结果值是一个不定长度的列表）
        # 每个索引对应着states中对应索引的长度
        env_lens = []
        # 遍历每一个游戏环境
        # 这一大段的循环作用是初始化环境，得到初始化结果
        for env in self.pool:
            # 每一次遍历时，重置游戏环境
            obs, obs_info = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # 如果支持矢量计算，那么可以将多个环境的输出结果，拼接到矩阵向量中，直接进行计算，效率比单个计算高
            if self.vectorized:
                # 矢量环境
                # 获取单词观察结果的向量长度
                obs_len = len(obs)
                # 将当前状态结果列表（应该是包含了环境状态，激励，动作，是否结束等信息）存储在states
                states.extend(obs)
            else:
                # 非矢量环境下，其观测的结果就是一个标量，简单说就是一个动作值，所以
                # 长度是1
                obs_len = 1
                # 将结果存储在status中
                states.append(obs)
            env_lens.append(obs_len)
            
            # 遍历本次环境观测的结果
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count)) # 创建对应环境观测结果历史缓存队列
                cur_rewards.append(0.0)  # 存储最初是状态下的激励，为0
                cur_steps.append(0) # 存储当前行走的部署，为0
                agent_states.append(self.agent.initial_state()) # 存储代理状态，reset环境时，代理状态是初始状态
        
        # 遍历索引
        # 从里这开始，应该就是尝试运行游戏了
        iter_idx = 0
        while True: 
            actions = [None] * len(states) # todo
            states_input = []
            states_indices = []
            # 遍历每一次的存储的观测状态
            # 对于非矢量环境来说，idx仅仅对应一个当前状态，但是对于矢量环境来说，idx对应当前获取的每个一观测值的索引
            # 这一个大循环的作用应该是，根据环境执行得到执行的动作结果
            # todo 有一个问题，矢量环境获取的多个观测结果这里怎么分辨
            for idx, state in enumerate(states):
                if state is None:
                    # 如果状态是空的，则使用环境进行随机选择一个动作执行， 另外这里假设的所有的环境都有相同的动作空间，
                    # 所以这里仅使用0索引环境进行随机动作采样
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    # 如果状态非空，则将当前的存储在states环境观测值存储在states_input中
                    # 并存储当前的索引
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                # 如果观测的状态列表非空，则将状态输入的神经网络环境代理中，获取将要执行的动作
                # 而agent_staes根据源码，发现并未做处理
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                # 遍历每一个状态所要执行的动作
                for idx, action in enumerate(states_actions):
                    # 获取当前动作对应的状态的索引位置，有上面106行的代码可知
                    g_idx = states_indices[idx]
                    # 将执行的动作存储在与状态相对应的索引上
                    actions[g_idx] = action
                    # 代理状态（todo 作用）
                    agent_states[g_idx] = new_agent_states[idx]
            # 将动作按照原先存储的每个环境得到的观测结果长度，按照环境数组进行重新分割分组
            grouped_actions = _group_list(actions, env_lens)
            
            # 因为存在一个大循环，存储每个环境的起始索引位置
            global_ofs = 0
            # 遍历每一个环境
            # 这一个大循环是将上一个循环中得到的执行动作应用到实际的环境中
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    # 这里action_n是一个list，也就是说矢量环境的输入的一个多维
                    # 如果是矢量的环境，则直接执行动作获取下一个状态，激励，是否结束等观测值
                    next_state_n, r_n, is_done_n, truncated, _ = env.step(action_n)
                    is_done_n = np.logical_or(is_done_n, truncated)
                else:
                    # 如果不是矢量环境，则需要将动作的第一个动作发送到env中获取相应的观测值（这里之所以是[0]，因为为了和矢量环境统一，即时是一个动作也会以列表的方式存储）
                    next_state, r, is_done, truncated, _ = env.step(action_n[0])
                    is_done = is_done or truncated
                    # 这个操作是为了和矢量环境统一
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]
                
                # 遍历每一次的动作所得到的下一个状态、激励、是否结束
                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # 获取当前缓存的索引位置
                    idx = global_ofs + ofs
                    # 获取初始环境的状态
                    # 因为action_n存储的就是每一个状态下所执行的动作，所以这里直接使用idx提取对应的状态
                    state = states[idx]
                    # 获取一个历史队列，此时队列为空
                    history = histories[idx]
                    agent_state = agent_states[idx]
                    agent_state.update(state, action, r, is_done, next_state)  # 更新代理状态
                    
                    # 这里利用的idx来区分每一个状态执行的动作所对应的激励值
                    # 将获取的激励值存储在缓存中
                    cur_rewards[idx] += r
                    # 将当前状态以及执行的动作，执行的步骤次数存在起来
                    cur_steps[idx] += 1
                    # 如果状态非空，则（当前状态，所执行的动作对应的历史缓存队列）将当前状态存储在history中
                    # 所以一个样本可能就是这样对应一个队列数据
                    if state is not None:
                        history.append((state, action, r, is_done, next_state, agent_state, env_idx))
                    # 如果达到了采集的步数并且遍历索引达到了两个经验样本的指定差值，则将样本返回，待外界下一次继续获取时，从这里继续执行
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    # 更新states，表示当前动作执行后状态的改变
                    # 将动作设置为动作执行后的下一个状态，因为idx表示当前运行的环境状态的变更
                    states[idx] = next_state
                    if is_done:
                        # 如果游戏结束，如果存储的历史数据小于指定的长度，则直接返回
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        # 弹出最左侧的历史数据，返回给外部获取数据
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        # 将当前状态+动作执行后得到的激励存储在total_rewards队列中
                        self.total_rewards.append(cur_rewards[idx])
                        # 这个当前状态+动作执行的次数也存储起来
                        self.total_steps.append(cur_steps[idx])
                        # 重置状态
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset()[0] if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                        
                # 将起始索引设置为下一个环境的起始位置
                global_ofs += len(action_n)
            # 遍历索引+1
            iter_idx += 1

    def pop_total_rewards(self):
        """
        返回所有采集的样本，并清空缓存
        """
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res




def _group_list(items, lens):
    """
    Unflat the list of items by lens
    反平铺队列，也就是将原先list 一维的数据，跟进lens进行分割，实现二维的队列
    [...] => [[.], [[.], [.]], .]
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
# state：当前的状态观测值
# action：当前状态下执行的动作
# reward：当前状态下获取的激励，如果是n步dqn，那么这个reward是n步下，省略了max q下得到的n步激励值和
# last_state: 下一个状态，如果是n步dqn，那么last_state表示第n步的下一个状态，用于计算最后一个状态的下一个状态的max q值
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    继承自ExperienceSource
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    这是一个围绕着ExperienceSource的包装器，以防止在我们需要的时候在回放缓冲区中存储完整的轨迹
    只有第一个和最后一个状态去查询完整的轨迹。对于每一个轨迹在计算激励时仅需要第一个状态、最后一个状态、在第一个状态时所执行的动作

    If we have partial trajectory at the end of episode, last_state will be None
    如果我们所获取的轨迹是包含结束，那么最后一个状态将会是None
    
    # todo steps_delta的作用
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        '''
        params steps_count: 这里的steps_count主要用于n步dqn的计算，提取最后一步的记录反馈和状态，所以虽然记录了step_count步，但是实际上遍历
         返回的还是仅当作执行一步
        '''
        # 判断参数类型、初始化父类、存储到成员变量
        assert isinstance(gamma, float)
        # 这里+1是因为，实际上在采样时，每次__iter__都只是获取当前执行完动作后当前的环境状态，而不知道下一个状态，所以需要+1，使得__iter__每次都能够返回2个采样，而
        # 最后一个采样就是下一个状态
        # todo 但是从代码中看到，这个当前状态实际上执行完step后才获取的，那么理论上这里的状态应该是下一个状态next_state，与实际需要获取的貌似不太匹配，需要查明原因
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        # 遍历当前环境获取的经验缓冲区
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            # 如果游戏结束，那么最后的状态设置为None
            # 返回的exp经验长度，会根据self.steps的长度不同而不同，这里的self.steps主要用于n步dqn的计算，提取最后一步的记录反馈和状态，如果是对于其他的方法，通常设置为1即可
            # 这个特性主要应用于N步dqn
            # 因为n步dqn的计算，需要利用到最后一个状态值计算q值，得到第n步的q值
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                # 获取最后一个经验的状态值
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            # 根据书中第120页的计算公式，计算bellman中，除了max q值的部分的激励值
            # 因为计算的时候省略中间步骤的max q操作
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)
            


class ExperienceSourceFirstLastRAW(ExperienceSourceRAW):
    """
    继承自ExperienceSource
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    这是一个围绕着ExperienceSource的包装器，以防止在我们需要的时候在回放缓冲区中存储完整的轨迹
    只有第一个和最后一个状态去查询完整的轨迹。对于每一个轨迹在计算激励时仅需要第一个状态、最后一个状态、在第一个状态时所执行的动作

    If we have partial trajectory at the end of episode, last_state will be None
    如果我们所获取的轨迹是包含结束，那么最后一个状态将会是None
    
    # todo steps_delta的作用
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        '''
        params steps_count: 这里的steps_count主要用于n步dqn的计算，提取最后一步的记录反馈和状态，所以虽然记录了step_count步，但是实际上遍历
         返回的还是仅当作执行一步
        '''
        # 判断参数类型、初始化父类、存储到成员变量
        assert isinstance(gamma, float)
        # 这里+1是因为，实际上在采样时，每次__iter__都只是获取当前执行完动作后当前的环境状态，而不知道下一个状态，所以需要+1，使得__iter__每次都能够返回2个采样，而
        # 最后一个采样就是下一个状态
        # todo 但是从代码中看到，这个当前状态实际上执行完step后才获取的，那么理论上这里的状态应该是下一个状态next_state，与实际需要获取的貌似不太匹配，需要查明原因
        super(ExperienceSourceFirstLastRAW, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        # 遍历当前环境获取的经验缓冲区
        for exp in super(ExperienceSourceFirstLastRAW, self).__iter__():
            # 如果游戏结束，那么最后的状态设置为None
            # 返回的exp经验长度，会根据self.steps的长度不同而不同，这里的self.steps主要用于n步dqn的计算，提取最后一步的记录反馈和状态，如果是对于其他的方法，通常设置为1即可
            # 这个特性主要应用于N步dqn
            # 因为n步dqn的计算，需要利用到最后一个状态值计算q值，得到第n步的q值
            if exp[-1][3] and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                # 获取最后一个经验的状态值
                last_state = exp[-1][0]
                elems = exp[:-1]
            total_reward = 0.0
            # 根据书中第120页的计算公式，计算bellman中，除了max q值的部分的激励值
            # 因为计算的时候省略中间步骤的max q操作
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e[2]
            yield (exp[0][0], exp[0][1], total_reward, exp[-1][3], last_state, exp[0][5])
            


def discount_with_dones(rewards, dones, gamma):
    '''
    根据传入的收集数据，计算折扣奖励类Q值，有点像PPO算法，也是累计未来的奖励到现在，用来预估一个很长的游戏轨迹最终会到哪一步

    param rewards: 传入游戏环境反馈的奖励
    param dones: 传入游戏是否结束的标识
    param gamma: 传入折扣因子

    return：返回计算得到的每步Q值
    '''

    discounted = []
    r = 0
    # 遍历奖励和结束标识
    # 计算累积折扣奖励，而这里的rewards[::-1], dones[::-1]遍历顺序是逆向的，因为是-1
    for reward, done in zip(rewards[::-1], dones[::-1]):
        # 根据获取到的奖励+上未来奖励（因为是逆向遍历）的折扣值，所以如果游戏没有结束，那么r会一直累计（有点像是计算Q值）
        # 但是如果遇到了游戏结束，那么就将重新开始计算奖励，不考虑未来的奖励
        r = reward + gamma*r*(1.-done)
        discounted.append(r)

    # 游戏累计的时候是逆向的，所以这里就再逆向一次，转换为正向的
    return discounted[::-1]


class ExperienceSourceRollouts:
    """
    N-step rollout experience source following A3C rollouts scheme. Have to be used with agent,
    keeping the value in its state (for example, agent.ActorCriticAgent).
    todo 我看讲解都是说是具备N步展开的经验池方法，但是这里的N步和其他的有什么区别吗？因为其他的也有steps_count参数

    Yields batches of num_envs * n_steps samples with the following arrays:
    1. observations
    2. actions
    3. discounted rewards, with values approximation
    4. values
    """
    def __init__(self, env, agent, gamma, steps_count=5):
        """
        Constructs the rollout experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions
        :param steps_count: 需要收集记录的游戏步数，并且不进行类似N步DQN的合并奖励操作，而是类似ppo一样直接返回连续的游戏轨迹
         这里就是和ExperienceSourceFirstLast的区别，这里的steps_count是指定的游戏步数，而不是n步dqn的计算
         todo 什么场景下使用这个方法
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(gamma, float)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()[0]) for e in self.pool] # states 是存储游戏环境的观测值
        mb_states = np.zeros((pool_size, self.steps_count) + states[0].shape, dtype=states[0].dtype) # 存储执行动作前的游戏状态
        mb_rewards = np.zeros((pool_size, self.steps_count), dtype=np.float32) # 存储执行动作后的激励
        mb_values = np.zeros((pool_size, self.steps_count), dtype=np.float32) # 存储执行动作后的Q值
        mb_actions = np.zeros((pool_size, self.steps_count), dtype=np.int64) # 存储执行的动作
        mb_dones = np.zeros((pool_size, self.steps_count), dtype=np.bool_) # 存储执行动作后游戏是否结束
        total_rewards = [0.0] * pool_size # 统计每个游戏环境的总激励（如果遇到游戏结束，则设置为0）
        total_steps = [0] * pool_size # 统计每个游戏执行到结束的总步数（如果遇到游戏结束，则设置为0）
        agent_states = None # 一开始代理状态为空（是Q值，也就是基于当前的状态预测下一个状态所能达到的最大Q值）
        step_idx = 0

        while True:
            actions, agent_states = self.agent(states, agent_states)
            rewards = []  # 存储执行动作后得到的奖励
            dones = [] # 存储执行动作后游戏是否结束
            new_states = [] # 存储执行动作后的游戏状态
            for env_idx, (e, action) in enumerate(zip(self.pool, actions)):
                '''
                遍历所有的游戏环境，执行动作，获取下一个状态，激励，是否结束
                这里应该只是执行一步
                '''
                o, r, done, truncated, _ = e.step(action)
                done = done or truncated
                total_rewards[env_idx] += r
                total_steps[env_idx] += 1
                if done:
                    o, o_info = e.reset()
                    self.total_rewards.append(total_rewards[env_idx])
                    self.total_steps.append(total_steps[env_idx])
                    total_rewards[env_idx] = 0.0
                    total_steps[env_idx] = 0
                new_states.append(np.array(o))
                dones.append(done)
                rewards.append(r)
            # we need an extra step to get values approximation for rollouts
            if step_idx == self.steps_count:
                # 如果游戏执行达到指定的步数，则遍历收集到的奖励
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, agent_states)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        # 在所有数据的尾部填充上已经得到的下一个状态的Q值，并将其结束的标识设置为False
                        env_rewards = discount_with_dones(env_rewards + [last_value], env_dones + [False], self.gamma)[:-1]
                    else:
                        # 处理达到步数后游戏结束的情况
                        env_rewards = discount_with_dones(env_rewards, env_dones, self.gamma)
                    # 根据转换后，mb_rewards已经被转换为了一种类似Q值的考虑到未来奖励的累计值
                    mb_rewards[env_idx] = env_rewards
                # 这里是将收集到的游戏状态进行展平操作，返回给外部，因为原始的数据第一维是环境的数量，第二维度是步数，第三维度以上才是收集到的数据
                # 为了更好的训练，需要展平，且此时也不需要游戏的状态了
                yield mb_states.reshape((-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), mb_values.flatten()
                step_idx = 0

            mb_states[:, step_idx] = states
            mb_rewards[:, step_idx] = rewards
            mb_values[:, step_idx] = agent_states
            mb_actions[:, step_idx] = actions
            mb_dones[:, step_idx] = dones
            step_idx += 1
            states = new_states

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        迭代器方法
        """
        while True:
            # 根据当前经验池的大小，创建一个集合
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


from collections import deque
# 经验重放缓冲区，主要用于收集训练样本，提取训练样本
class ExperienceReplayChunkBuffer:
    def __init__(self, experience_source, buffer_size, bit_depth=32):
        '''
        buffer有固定大小，并且每次sample时采样batch_size个chunk_size连续片段
        '''
        
        assert isinstance(buffer_size, int)
        # 将经验池转换为迭代器
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        # 重放缓冲区的大小
        self.capacity = buffer_size
        # 当前遍历的位置
        self.pos = 0
        self.full = False
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth
        self.size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 迭代器不能被pickle保存，所以移除它
        state['experience_source_iter'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 加载后需要手动调用 set_exp_source() 恢复经验池
        self.experience_source_iter = None
    

    def _sample_idx(self, L):
        while True:
            idx = np.random.randint(0, self.size if self.full else self.pos - L)
            idxs = np.arange(idx, idx + L) % self.size
            if self.pos not in idxs[1:]:  # 确保数据不跨越内存索引
                break

        return idxs

    def sample(self, batch_size, chunk_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size or len(self.buffer) < chunk_size:
            return self.buffer

        sample_data_list = [[self.buffer[i] for i in self._sample_idx(chunk_size).tolist()] for _ in range(batch_size)]
        # Warning: replace=False makes random.choice O(n)
        return sample_data_list

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample

        if (self.pos + 1) >= self.capacity:
            self.full = True
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer 提取样本到重放缓存区中
        :param samples: how many samples to populate  从样本池中提取多少个样本到缓冲区
        
        算法的原理及利用迭代器根据数量，从经验池中获取数据
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


    def set_exp_source(self, experience_source):
        self.experience_source_iter = iter(experience_source)


# 经验重放缓冲区，主要用于收集训练样本，提取训练样本
class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        '''
        param experience_source: 经验池
        param buffer_size: 每次提取的样本大小
        '''
        
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        # 将经验池转换为迭代器
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        # 重放缓冲区的大小
        self.capacity = buffer_size
        # 当前遍历的位置
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer 提取样本到重放缓存区中
        :param samples: how many samples to populate  从样本池中提取多少个样本到缓冲区
        
        算法的原理及利用迭代器根据数量，从经验池中获取数据
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

class PrioReplayBufferNaive:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def __getstate__(self):
        state = self.__dict__.copy()
        # 迭代器不能被pickle保存，所以移除它
        state['experience_source_iter'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience
    batch to form suitable to learning.
    """
    def preprocess(self, batch):
        raise NotImplementedError


class QLearningPreprocessor(BatchPreprocessor):
    """
    Supports SimpleDQN, TargetDQN, DoubleDQN and can additionally feed TD-error back to
    experience replay buffer.

    To use different modes, use appropriate class method
    """
    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99, device="cpu"):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma
        self.device = device

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states_first, states_last):
        """
        Calculates apropriate q values for first and last states. Way of calculate depends on our settings.
        :param states_first: numpy array of first states
        :param states_last: numpy array of last states
        :return: tuple of numpy arrays of q values
        """
        # here we need both first and last values calculated using our main model, so we
        # combine both states into one batch for efficiency and separate results later
        if self.target_model is None or self.use_double_dqn:
            states_t = torch.tensor(np.concatenate((states_first, states_last), axis=0)).to(self.device)
            res_both = self.model(states_t).data.cpu().numpy()
            return res_both[:len(states_first)], res_both[len(states_first):]

        # in this case we have target_model set and use_double_dqn==False
        # so, we should calculate first_q and last_q using different models
        states_first_v = torch.tensor(states_first).to(self.device)
        states_last_v = torch.tensor(states_last).to(self.device)
        q_first = self.model(states_first_v).data
        q_last = self.target_model(states_last_v).data
        return q_first.cpu().numpy(), q_last.cpu().numpy()

    def _calc_target_rewards(self, states_last, q_last):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states_last: numpy array of last states from the games
        :param q_last: numpy array of last q values
        :return: vector of target rewards
        """
        # in this case we handle both simple DQN and target DQN
        if self.target_model is None or not self.use_double_dqn:
            return q_last.max(axis=1)

        # here we have target_model set and use_double_dqn==True
        actions = q_last.argmax(axis=1)
        # calculate Q values using target net
        states_last_v = torch.tensor(states_last).to(self.device)
        q_last_target = self.target_model(states_last_v).data.cpu().numpy()
        return q_last_target[range(q_last_target.shape[0]), actions]

    def preprocess(self, batch):
        """
        Calculates data for Q learning from batch of observations
        :param batch: list of lists of Experience objects
        :return: tuple of numpy arrays:
            1. states -- observations
            2. target Q-values
            3. vector of td errors for every batch entry
        """
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0, qL = self._calc_Q(state_0, state_L)
        rewards = self._calc_target_rewards(state_L, qL)

        td = np.zeros(shape=(len(batch),))

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final reward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward and calculate td error
            act = exps[0].action
            td[idx] = q0[idx][act] - total_reward
            q0[idx][act] = total_reward

        return state_0, q0, td


ExperienceNextStates = namedtuple('ExperienceNextStates', ['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceSourceNextStates:
    """
    Simple n-step experience source using single or multiple environments
    简单的存储n步的经验采集样本，用于单个或者多个环境中

    Every experience contains n list of ExperienceNextStates entries
    每个经验样本集都包含n步的经验
    """

    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used 环境信息或者环境信息列表
        :param agent: callable to convert batch of states into actions to take 代理信息
        :param steps_count: count of steps to track for every experience chain 需要追溯多少步以前的记录
        :param steps_delta: how many steps to do between experience items todo
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        # 判断经验传入的参数类型是否正确
        # 并存储到成员变量
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        # self.pool: 存储游戏环境
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        # 这个接口就是运行环境并获取观测值，填充经验缓冲区的地方
        # 调用这个接口后，每次遍历都会使用神经网络预测当前状态下的
        # 执行动作，使用yield实现
        # 如果游戏一旦结束，将会把最后一次的游戏状态存储到total_rewards（包括执行的动作，奖励值，游戏的环境状态）
        #
        # return:
        #

        # states: 存储每一次的环境观测值
        # agent_states: 存储游戏网络的代理的初始状态
        # histories: 存储的好像是一个队列，长度为step_ount，存储的应该是历史记录，存储多少步以前的状态等信息 todo
        # cur_rewards: 存储每轮游戏观测的激励，游戏初始化时存储的是0.0
        # cur_steps: 存储当前观测结果的步数（todo），游戏初始化时存储的是0
        # agent_states： 代理状态, 作用1：表示当前游戏主体所处状态的评价值（Q值） todo 作用
        # 看代码后，发现这个agnet states的状态只有在ExperienceSourceRollouts中才有实际的作用

        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        # 存储每次环境观测值的长度（因为矢量的环境其返回的结果值是一个不定长度的列表）
        # 每个索引对应着states中对应索引的长度
        env_lens = []
        # 遍历每一个游戏环境
        # 这一大段的循环作用是初始化环境，得到初始化结果
        for env in self.pool:
            # 每一次遍历时，重置游戏环境
            obs, obs_info = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # 如果支持矢量计算，那么可以将多个环境的输出结果，拼接到矩阵向量中，直接进行计算，效率比单个计算高
            if self.vectorized:
                # 矢量环境
                # 获取单词观察结果的向量长度
                obs_len = len(obs)
                # 将当前状态结果列表（应该是包含了环境状态，激励，动作，是否结束等信息）存储在states
                states.extend(obs)
            else:
                # 非矢量环境下，其观测的结果就是一个标量，简单说就是一个动作值，所以
                # 长度是1
                obs_len = 1
                # 将结果存储在status中
                states.append(obs)
            env_lens.append(obs_len)

            # 遍历本次环境观测的结果
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))  # 创建对应环境观测结果历史缓存队列
                cur_rewards.append(0.0)  # 存储最初是状态下的激励，为0
                cur_steps.append(0)  # 存储当前行走的部署，为0
                agent_states.append(self.agent.initial_state())  # 存储代理状态，reset环境时，代理状态是初始状态

        # 遍历索引
        # 从里这开始，应该就是尝试运行游戏了
        iter_idx = 0
        while True:
            actions = [None] * len(states)  # todo
            states_input = []
            states_indices = []
            # 遍历每一次的存储的观测状态
            # 对于非矢量环境来说，idx仅仅对应一个当前状态，但是对于矢量环境来说，idx对应当前获取的每个一观测值的索引
            # 这一个大循环的作用应该是，根据环境执行得到执行的动作结果
            # todo 有一个问题，矢量环境获取的多个观测结果这里怎么分辨
            for idx, state in enumerate(states):
                if state is None:
                    # 如果状态是空的，则使用环境进行随机选择一个动作执行， 另外这里假设的所有的环境都有相同的动作空间，
                    # 所以这里仅使用0索引环境进行随机动作采样
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    # 如果状态非空，则将当前的存储在states环境观测值存储在states_input中
                    # 并存储当前的索引
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                # 如果观测的状态列表非空，则将状态输入的神经网络环境代理中，获取将要执行的动作
                # 而agent_staes根据源码，发现并未做处理
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                # 遍历每一个状态所要执行的动作
                for idx, action in enumerate(states_actions):
                    # 获取当前动作对应的状态的索引位置，有上面106行的代码可知
                    g_idx = states_indices[idx]
                    # 将执行的动作存储在与状态相对应的索引上
                    actions[g_idx] = action
                    # 代理状态（todo 作用）
                    agent_states[g_idx] = new_agent_states[idx]
            # 将动作按照原先存储的每个环境得到的观测结果长度，按照环境数组进行重新分割分组
            grouped_actions = _group_list(actions, env_lens)

            # 因为存在一个大循环，存储每个环境的起始索引位置
            global_ofs = 0
            # 遍历每一个环境
            # 这一个大循环是将上一个循环中得到的执行动作应用到实际的环境中
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    # 这里action_n是一个list，也就是说矢量环境的输入的一个多维
                    # 如果是矢量的环境，则直接执行动作获取下一个状态，激励，是否结束等观测值
                    next_state_n, r_n, is_done_n, truncated, _ = env.step(action_n)
                    is_done_n = is_done_n or truncated
                else:
                    # 如果不是矢量环境，则需要将动作的第一个动作发送到env中获取相应的观测值（这里之所以是[0]，因为为了和矢量环境统一，即时是一个动作也会以列表的方式存储）
                    next_state, r, is_done, truncated, _ = env.step(action_n[0])
                    is_done = is_done or truncated
                    # 这个操作是为了和矢量环境统一
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                # 遍历每一次的动作所得到的下一个状态、激励、是否结束
                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # 获取当前缓存的索引位置
                    idx = global_ofs + ofs
                    # 获取初始环境的状态
                    # 因为action_n存储的就是每一个状态下所执行的动作，所以这里直接使用idx提取对应的状态
                    state = states[idx]
                    # 获取一个历史队列，此时队列为空
                    history = histories[idx]

                    # 这里利用的idx来区分每一个状态执行的动作所对应的激励值
                    # 将获取的激励值存储在缓存中
                    cur_rewards[idx] += r
                    # 将当前状态以及执行的动作，执行的步骤次数存在起来
                    cur_steps[idx] += 1
                    # 如果状态非空，则（当前状态，所执行的动作对应的历史缓存队列）将当前状态存储在history中
                    # 所以一个样本可能就是这样对应一个队列数据
                    if state is not None:
                        history.append(ExperienceNextStates(state=state, action=action, reward=r, done=is_done, next_state=next_state))
                    # 如果达到了采集的步数并且遍历索引达到了两个经验样本的指定差值，则将样本返回，待外界下一次继续获取时，从这里继续执行
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    # 更新states，表示当前动作执行后状态的改变
                    # 将动作设置为动作执行后的下一个状态，因为idx表示当前运行的环境状态的变更
                    states[idx] = next_state
                    if is_done:
                        # 如果游戏结束，如果存储的历史数据小于指定的长度，则直接返回
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        # 弹出最左侧的历史数据，返回给外部获取数据
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        # 将当前状态+动作执行后得到的激励存储在total_rewards队列中
                        self.total_rewards.append(cur_rewards[idx])
                        # 这个当前状态+动作执行的次数也存储起来
                        self.total_steps.append(cur_steps[idx])
                        # 重置状态
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset()[0] if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()

                # 将起始索引设置为下一个环境的起始位置
                global_ofs += len(action_n)
            # 遍历索引+1
            iter_idx += 1

    def pop_total_rewards(self):
        """
        返回所有采集的样本，并清空缓存
        """
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
    

class A2CExperienceSourceDirect:
    def __init__(self, env, agent, gamma, steps_count=5):
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(gamma, float)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()[0]) for e in self.pool] # states 是存储游戏环境的观测值
        mb_states = np.zeros((pool_size, self.steps_count) + states[0].shape, dtype=states[0].dtype) # 存储执行动作前的游戏状态
        mb_rewards = np.zeros((pool_size, self.steps_count), dtype=np.float32) # 存储执行动作后的激励
        mb_values = np.zeros((pool_size, self.steps_count), dtype=np.float32) # 存储执行动作后的Q值
        mb_actions = np.zeros((pool_size, self.steps_count), dtype=np.int64) # 存储执行的动作
        mb_dones = np.zeros((pool_size, self.steps_count), dtype=np.bool_) # 存储执行动作后游戏是否结束
        total_rewards = [0.0] * pool_size # 统计每个游戏环境的总激励（如果遇到游戏结束，则设置为0）
        total_steps = [0] * pool_size # 统计每个游戏执行到结束的总步数（如果遇到游戏结束，则设置为0）
        agent_states = self.agent.initial_state() # 初始化代理状态
        step_idx = 0

        mb_hx = np.zeros((self.steps_count,) + agent_states[0].shape, dtype=np.float32)  # 存储每个游戏环境的代理状态
        mb_cx = np.zeros((self.steps_count,) + agent_states[1].shape, dtype=np.float32)  # 存储每个游戏环境的代理状态


        while True:
            actions, values, new_agent_states = self.agent(states, agent_states)
            rewards = []  # 存储执行动作后得到的奖励
            dones = [] # 存储执行动作后游戏是否结束
            new_states = [] # 存储执行动作后的游戏状态
            for env_idx, (e, action) in enumerate(zip(self.pool, actions)):
                '''
                遍历所有的游戏环境，执行动作，获取下一个状态，激励，是否结束
                这里应该只是执行一步
                '''
                o, r, done, truncated, _ = e.step(action)
                done = done or truncated
                total_rewards[env_idx] += r
                total_steps[env_idx] += 1
                if done:
                    o, o_info = e.reset()
                    self.total_rewards.append(total_rewards[env_idx])
                    self.total_steps.append(total_steps[env_idx])
                    total_rewards[env_idx] = 0.0
                    total_steps[env_idx] = 0
                    new_agent_states[0][:, env_idx, :], new_agent_states[1][:, env_idx, :] = 0, 0
                new_states.append(np.array(o))
                dones.append(done)
                rewards.append(r)
            # we need an extra step to get values approximation for rollouts
            if step_idx == self.steps_count:
                # 如果游戏执行达到指定的步数，则遍历收集到的奖励
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, values)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        # 在所有数据的尾部填充上已经得到的下一个状态的Q值，并将其结束的标识设置为False
                        env_rewards = discount_with_dones(env_rewards + [last_value], env_dones + [False], self.gamma)[:-1]
                    else:
                        # 处理达到步数后游戏结束的情况
                        env_rewards = discount_with_dones(env_rewards, env_dones, self.gamma)
                    # 根据转换后，mb_rewards已经被转换为了一种类似Q值的考虑到未来奖励的累计值
                    mb_rewards[env_idx] = np.array(env_rewards).flatten()
                # 这里是将收集到的游戏状态进行展平操作，返回给外部，因为原始的数据第一维是环境的数量，第二维度是步数，第三维度以上才是收集到的数据
                # 为了更好的训练，需要展平，且此时也不需要游戏的状态了
                yield mb_states.reshape((-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), mb_values.flatten(), np.transpose(mb_hx, (1, 0, 2, 3)).reshape(mb_hx.shape[1], -1, mb_hx.shape[3]), np.transpose(mb_cx, (1, 0, 2, 3)).reshape(mb_cx.shape[1], -1, mb_cx.shape[3])
                step_idx = 0

            mb_states[:, step_idx] = states
            mb_rewards[:, step_idx] = rewards
            mb_values[:, step_idx] = values.squeeze(1)
            mb_actions[:, step_idx] = actions
            mb_dones[:, step_idx] = dones
            mb_hx[step_idx] = agent_states[0]
            mb_cx[step_idx] = agent_states[1]
            step_idx += 1
            states = new_states
            agent_states = new_agent_states

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class ExperienceEpisodeeplayBuffer:
    def __init__(self, experience_source, epsilon_size, exisode_length=300, d_type=torch.float32, device='cpu'):
        '''
        self.buffer中每个元素是一个完整的episode周期，里面存储的是每个episode的经验样本
        '''
        
        # 将经验池转换为迭代器
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = deque(maxlen=epsilon_size)
        self.episode = []
        # 当前遍历的位置
        self.pos = 0
        self.d_type = d_type
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 迭代器不能被pickle保存，所以移除它
        state['experience_source_iter'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 加载后需要手动调用 set_exp_source() 恢复经验池
        self.experience_source_iter = None


    def sample(self, batch_size, chunk_length):
        '''
        这里传入的batch_size，采样batch_size个样本数据
        '''
        # 随机选择batch_size个episode
        sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        # chunked_episodes看起来也是一个连续的片段
        chunked_episodes = list()
        for ep_idx in sampled_indices:
            # 随机选择一个起始位置
            start_idx = np.random.randint(low=0, high=len(self.buffer[ep_idx]) - chunk_length)
            # 将选择连续的chunk_length个数据添加到chunked_episodes中
            chunked_episodes.append(self.buffer[ep_idx][start_idx:start_idx + chunk_length])
        # 此时chunked_episodes是一个list，里面存储的是每个episode的连续片段
        serialized_episodes = self.serialize_episode(chunked_episodes)
        return serialized_episodes

    def _add(self, sample):
        if sample[0][3]:
            self.buffer.append(self.episode)
            self.episode = []
        else:
            self.episode.append(sample)


    def populate(self, samples):
        """
        Populates samples into the buffer 提取样本到重放缓存区中
        :param samples: how many samples to populate  从样本池中提取多少个样本到缓冲区
        
        算法的原理及利用迭代器根据数量，从经验池中获取数据
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

    
    def serialize_episode(self, list_episodes):
        '''
        return 返回一个字典，里面存储的是每个episode的observation、action、reward
        '''

        batched_ep_obs, batched_ep_action, batched_ep_reward = list(), list(), list()
        # 拆分每个episode的observation、action、reward到独立的缓存list中，list的每个元素是一个连续的obs、action、erward tensor
        for episode in list_episodes:
            ep_obs = torch.from_numpy(np.array([transition[0][0] for transition in episode]))
            ep_action = torch.from_numpy(np.array([transition[0][1] for transition in episode]))
            ep_reward = torch.from_numpy(np.array([transition[0][2] for transition in episode]))
            batched_ep_obs.append(ep_obs)
            batched_ep_action.append(ep_action)
            batched_ep_reward.append(ep_reward)
        # 最后将所有的list转换为tensor
        # batched_ep_obs shape 是（batch_size， chunk_length， visual_resolution， visual_resolution， channels）
        # batched_ep_action shape 是（batch_size， chunk_length， action_dim）
        # batched_ep_reward shape 是（batch_size， chunk_length， 1）
        batched_ep_obs = torch.stack(batched_ep_obs).to(self.d_type).to(self.device)
        batched_ep_action = torch.stack(batched_ep_action).to(self.d_type).to(self.device)
        batched_ep_reward = torch.stack(batched_ep_reward).to(self.d_type).to(self.device)
        # batched_ep_obs shape 是（chunk_length， batch_size， visual_resolution， visual_resolution， channels）
        # batched_ep_action shape 是（chunk_length， batch_size， action_dim）
        # batched_ep_reward shape 是（chunk_length， batch_size， 1）
        # 这里的转换是为了将时间步放在前面，方便后续的训练
        return {'obs': batched_ep_obs.transpose(0, 1), 'action': batched_ep_action.transpose(0, 1), 'reward': batched_ep_reward.transpose(0, 1)}
