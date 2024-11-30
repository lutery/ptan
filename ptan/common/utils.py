import sys
import time
import operator
from datetime import timedelta
import numpy as np
import collections

import torch
import torch.nn as nn


class SMAQueue:
    """
    Queue of fixed size with mean, max, min operations
    """
    def __init__(self, size):
        self.queue = collections.deque()
        self.size = size

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            self.queue.extend(other)
        else:
            self.queue.append(other)
        while len(self.queue) > self.size:
            self.queue.popleft()
        return self

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "SMAQueue(size=%d)" % self.size

    def __str__(self):
        return "SMAQueue(size=%d, len=%d)" % (self.size, len(self.queue))

    def min(self):
        if not self.queue:
            return None
        return np.min(self.queue)

    def mean(self):
        if not self.queue:
            return None
        return np.mean(self.queue)

    def max(self):
        if not self.queue:
            return None
        return np.max(self.queue)


class SpeedMonitor:
    def __init__(self, batch_size, autostart=True):
        self.batch_size = batch_size
        self.start_ts = None
        self.batches = None
        if autostart:
            self.reset()

    def epoch(self):
        if self.epoches is not None:
            self.epoches += 1

    def batch(self):
        if self.batches is not None:
            self.batches += 1

    def reset(self):
        self.start_ts = time.time()
        self.batches = 0
        self.epoches = 0

    def seconds(self):
        """
        Seconds since last reset
        :return:
        """
        return time.time() - self.start_ts

    def samples_per_sec(self):
        """
        Calculate samples per second since last reset() call
        :return: float count samples per second or None if not started
        """
        if self.start_ts is None:
            return None
        secs = self.seconds()
        if abs(secs) < 1e-5:
            return 0.0
        return (self.batches + 1) * self.batch_size / secs

    def epoch_time(self):
        """
        Calculate average epoch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.epoches > 0:
            s /= self.epoches + 1
        return timedelta(seconds=s)

    def batch_time(self):
        """
        Calculate average batch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.batches > 0:
            s /= self.batches + 1
        return timedelta(seconds=s)


class WeightedMSELoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        if weights is None:
            return nn.MSELoss(self.size_average)(input, target)

        loss_rows = (input - target) ** 2
        if len(loss_rows.size()) != 1:
            loss_rows = torch.sum(loss_rows, dim=1)
        res = (weights * loss_rows).sum()
        if self.size_average:
            res /= len(weights)
        return res


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind

    TensorBoard值跟踪器，用来记录训练过程中的状态值，特点是仅记录最近的10个状态值
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        '''
        在with as语法下生效，此时会创建一个字典类型，字典存储的value类型为list
        '''
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        '''
        param param_name: 保存的记录表名，相同的名称会记录到同一张表中
        param value： 记录值（Y轴）
        param iter_index: 第几轮（X轴）
        '''
        
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        # 从batches中读取指定类型的数据集合
        # 将数据保存到data中
        data = self._batches[param_name]
        data.append(self._as_float(value))

        # 如果数据超过了执行的长度，则将记录的数据写入到tensorboard中，并清空数据
        # 写入到tensorboard中的记录值是平均值
        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


class RewardTracker:
    def __init__(self, writer, min_ts_diff=1.0, info_callback=None):
        """
        Constructs RewardTracker
        :param writer: writer to use for writing stats,传入SummaryWriter类型，用来跟踪训练中的奖励
        :param min_ts_diff: minimal time difference to track speed 最小的时间差跟踪速度，奖励记录是仅记录一个时间段的奖励值，当当前时间到达这个时间差min_ts_diff后，就会重新记录起始时间以及当前进行的帧数
        """
        self.writer = writer
        self.min_ts_diff = min_ts_diff
        self.info_callback = info_callback

    def __enter__(self):
        '''
        with as 语法开始时会执行这里
        '''
        self.ts = time.time()
        self.ts_frame = 0 # 记录上次时间
        self.total_rewards = [] # 清空目前为止所有得到的奖励
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        '''
        param reward: 奖励
        param frame: 训练过程中总共进行的帧数
        '''
        self.total_rewards.append(reward) # 存储奖励
        mean_reward = np.mean(self.total_rewards[-100:]) # 记录最新的100个奖励的平均值
        ts_diff = time.time() - self.ts # 计算上次重置时间开始经过的的时间差
        if ts_diff > self.min_ts_diff: # 如果时间差超过了指定的大小，
            speed = (frame - self.ts_frame) / ts_diff # 计算当前时间段类的进行的帧数，用来计算游戏的速度
            self.ts_frame = frame # 重置起始第几帧
            self.ts = time.time() # 重置起始记录时间
            epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon # todo 作用
            print("%d: done %d episodes, mean reward %.3f, speed %.2f f/s%s" % (
                frame, len(self.total_rewards), mean_reward, speed, epsilon_str
            )) # 打印帧数、记录的总奖励个数，最新100次游戏的平均奖励，游戏进行的速度，todo epsilon的作用
            if self.info_callback:
                self.info_callback("%d: done %d episodes, mean reward %.3f, speed %.2f f/s%s" % (
                frame, len(self.total_rewards), mean_reward, speed, epsilon_str
            ))
            sys.stdout.flush()
            self.writer.add_scalar("speed", speed, frame) # 记录速度
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame) # 记录epsilon
        self.writer.add_scalar("reward_100", mean_reward, frame) # 记录最近100游戏的平均奖励
        self.writer.add_scalar("reward", reward, frame) # 记录最新一次游戏的奖励
        return mean_reward if len(self.total_rewards) > 30 else None # 如果记录的总奖励数超过30个，则返回平均奖励值否则返回None
