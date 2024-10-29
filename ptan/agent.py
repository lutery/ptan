"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from . import actions


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    这个预处理器的方法就是将list转换为矩阵的形式
    如果state是一维的，那么就将其转换为[1, D]的形式
    如果state是多维的，那么就将其转换为[N, E, D]的形式
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.asarray([np.asarray(s) for s in states])
    return torch.tensor(np_states)


def float32_preprocessor(states):
    '''
    将状态矩阵转换为float32数值类型的tensor
    '''
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQN代理 是一个无内存记忆的DQN网络，不存储其他推理的数据
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        '''
        param dqn_model: dqn网络模型，训练的的网络
        param action_selector: 动作选择器
        '''
        
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        # 从给定的一系列states中，选择将要执行的动作值
        # 而agent_states并没有参与计算，仅仅只是保证如果是None则返回和states维度一样的None列表
        # 如果非None，则返回原值，作用未知
        if agent_states is None:
            agent_states = [None] * len(states)
        # 如果定义了预处理器，则将state进行预处理
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        # 用传入的模型计算预测q值或者动作概率
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states

# 目标网络
class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        # 训练的网络模型
        self.model = model
        # 拷贝一份，作为目标网路模型存储
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    # TODO: unify code with DQNAgent, as only action selector is differs.
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        '''
            model: 策略动作推理网络
            preprocessor: 将计算的结果转换的数据类型，比如转换为float32
            apply_softmax: 使用对model的计算结果使用softmax计算结果
        '''
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states 在本代理器中，agent_states没有参与计算，仅仅是保证其维度和states一样
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        # 如果定义了预处理器，则进行预处理擦欧总
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        # 计算动作概率
        probs_v = self.model(states)
        # 如果需要使用softmax计算
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        # 将网络得到的动作概率丢给动作选择器进行选择需要执行的动作
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        '''
        param model: 策略网络
        param action_selector: 动作选择器，默认值是根据预测的动作概率分布选择动作
        param device: 计算设备
        param apply_softmax: 是否使用softmax计算动作概率
        param preprocessor: 预处理器，将状态转换为float32类型
        '''
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        看了代码，其和PolicyAgent不同的地方就是对于状体的处理
        Return actions from given list of states
        :param states: list of states 返回此时环境状态的评价值也是游戏主体本身所处的状态的评价值
        :return: list of actions
        """
        # 将数据进行预处理，不改变数据本身，进转换为tensor
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        # 得到动作概率，选择执行的动作
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states
