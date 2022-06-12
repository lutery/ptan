import numpy as np
from typing import Union


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    # epsilon动作选择器
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        # 原理是根据网路推理出现的每个动作所执行后得到的q值，
        # 根据每个动作的q值选择将要执行的动作
        # 再根据epsilon决定是否需要将动作用随机动作替代
        # 最后返回选择的动作
        # parma scores: q值 shape(batch_size, actions)
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        # actions shape is （batch_size, 1），其中1表示所选择动作索引
        actions = self.selector(scores)
        # 下面的对比操作，所有小于sel.epsilon的都为True，所以大于self.epsilon都为true
        # 结果会类似于[true false false...]
        # 意义：在本次批量训练的样本中，随机选择一些样本使用随机探索的方式进行执行动作选择
        mask = np.random.random(size=batch_size) < self.epsilon
        # sum(mask)统计有多少个动作是随机选择
        # np.random.choice表示每个需要随机的动作选择所执行的动作的索引
        rand_actions = np.random.choice(n_actions, sum(mask))
        # 使用这种方式赋值，可以保证仅赋值给为True索引的数组，
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)
