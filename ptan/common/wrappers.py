"""basic wrappers, useful for reinforcement learning on gym envs"""
# Mostly copy-pasted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
import cv2


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        重置时随机选择无操作动作的次数，根据针对的应该是多条生命的游戏
        """
        super(NoopResetEnv, self).__init__(env)
        # 无操作重置的次数
        self.noop_max = noop_max
        # 是否覆盖随机值
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(seed=seed, options=options)
        # 选择无操作执行动作的次数
        if self.override_num_noops is not None:
            # 固定值
            noops = self.override_num_noops
        else:
            # 随机值
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        info = {}
        # 无操作执行动作有可能游戏会结束，但这里检测到之后，会继续调用reset
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(0)
            if done:
                obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        # 以下可知，一些游戏存在FIRE的动作，并且存在FIRE动作的游戏其游戏动作执行有三个以上
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 这里之所以尝试重置后尝试各种动作，是因为不知道哪个是FIRE，继续游戏，所以一个一个尝试
        # 如果不小心游戏结束了，则继续重置
        # 假设游戏继续游戏的按钮在前3
        self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset(seed=seed, options=options)
        obs, _, done, _, info = self.env.step(2)
        if done:
            self.env.reset(seed=seed, options=options)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    # 这个环境包装器是针对有多条生命的游戏实现的
    # 因为代码中所实现的训练应该都是以一条命来实现的，所以这里为了避免专门针对多条生命进行实现特例，将
    # 具备多条生命的游戏也包装成单条生命
    '''
    游戏中的角色可能有多个生命。在这样的游戏中，通常当角色失去一条生命时，并不意味着游戏结束，只有当所有的生命都耗尽时，游戏才真正结束。这种设计的目的是使得深度强化学习算法能够更好地估算值函数，因为在许多情况下，失去一条生命和游戏结束在策略上可能具有相同的影响
    '''
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        # 记录游戏角色的生命数量。
        self.lives = 0
        # 指示在上一个步骤中游戏是否真正结束
        self.was_real_done = True
        # 指示环境是否真正重置
        self.was_real_reset = False

    def step(self, action):
        # 指定动作
        obs, reward, done, truncated, info = self.env.step(action)
        # 记录动作的结果，如果游戏具备多条生命，就算损失了一条生命，这里的done依旧是false
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        # 获取游戏的剩余生命数
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # 如果剩余生命数和上一次指定动作不同，则说明损失了一条生命
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            # 标识游戏已经结束，并返回，将具备单条生命的游戏模拟成单条生命
            done = True
        # 保存生命数
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        # 如果游戏真的，则直接调用环境的reset方法
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            # 如果游戏没有结束，则不执行任何动作，模拟重置
            obs, _, _, _, info = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        # 返回游戏状态
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    # 这里会将传入的游戏动作重复执行，直到达到skip次
    # 该包装器的目的是:
# 跳过一些帧，即每次不执行每一帧，而是执行每skip帧，这可以加速学习过程，因为在连续的几帧中，游戏的状态可能变化不大。
# 使用最大池化技术来选择两个连续帧之间的最大像素值。这是为了解决Atari游戏的闪烁问题，其中某些对象可能不会出现在每一帧中。
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2) # 最多保存2帧的观测数据，用于最大池化
        # 执行动作的次数
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        truncated = None
        # 重复执行相同的动作skip次
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            # 存储最近的maxLen次观测值
            self._obs_buffer.append(obs)
            # 累计总奖励
            total_reward += reward
            if done:
                #如果游戏结束则跳出循环
                break

        # 将多次观测到的游戏环境组合成一次观测值
        # 即将游戏的多帧组合成一帧，这里相同的是两帧组合成一帧
        # 使用最大池化技术（通过np.max）来合并_obs_buffer中的最后两帧。这有助于解决Atari游戏的某些对象可能在连续的帧中闪烁的问题
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Clear past frame buffer and init. to first obs. from inner env."""
        # 重置，清空缓存
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """
    将游戏画面（观察空间）转换为84*84的灰度图片
    """
    
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        # 创建新的观察空间，值范围0~255的单通道（84*84）尺寸的图片
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """
        将观察状态进行转换
        """
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        # 根据不同的帧尺寸进行转换为灰度单通道84，118的帧
        # 然后直接截取高度中间的部分，作为84*84的图片显示
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ClippedRewardsWrapper(gym.RewardWrapper):
    """
    通过使用Numpy sign, 将激励限制在-1~1之间
    """
    
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was.
        该对象确保观察之间的公共帧只存储一次。它的存在纯粹是为了优化内存使用，这对于DQN的1M帧重放来说是巨大的
        缓冲区。
        该对象在传递给模型之前应该只被转换为numpy数组。
        你不会相信之前的解有多复杂
        """
        self._frames = frames

    def __array__(self, dtype=None):
        # 将多帧连接后，返回
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """帧堆栈，通过这个保证每次返回的观察空间结果仅具备K帧组合
        这种方式，主要用于连续帧的游戏，因为只有连续帧才能很好的判断出游戏的需要执行的动作，否则不知道当前游戏的球飞的方向
        Stack k last frames.
        仅存储最新的K帧
        Returns lazy array, which is much more memory efficient.
        通过返回LazyFrame，优化内存的使用
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        # 创建一个最大只有K帧的队列，一旦长度超过K，那么自动将队头的帧去除
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # 因为k帧进行链接，所以观察空间的shape就变成了(shp[0]*k, shp[1], shp[2])
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 在重置的时候，因为只有一帧，所以进行这一帧复制K份进行存储
        # 在通过get_ob将这K帧组合后返回
        ob, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    将图片转换为CWH的格式，适用于pytorch计算格式
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_dqn(env, stack_frames=4, episodic_life=True, reward_clipping=True):
    """
    Apply a common set of wrappers for Atari games.
    扩展dqn环境
    param env:
    param stack_frames:
    param episodic_life: 是否创建基于血量值的环境
    param reward_clipping:
    """
    assert 'NoFrameskip' in env.spec.id
    if episodic_life:
        # 将多条生命的游戏模拟成单条生命
        env = EpisodicLifeEnv(env)
    # 增强初始化
    env = NoopResetEnv(env, noop_max=30)
    # 跳帧包装器
    env = MaxAndSkipEnv(env, skip=4)
    '''
    这个判断语句检查环境中是否存在一个名为"FIRE"的动作。具体来说，它检查gym环境提供的动作列表中是否包含"FIRE"这个动作。
    这在某些Atari游戏中是很有用的。例如，在游戏开始或角色失去一条命后，许多Atari游戏需要玩家按"FIRE"键来重新开始或继续游戏。在自动化这些游戏的强化学习设置中，如果玩家（代理）不发送"FIRE"动作，游戏可能根本不会开始，因此代理将无法学习。
    通过这个判断，可以确保在需要的情况下自动发送"FIRE"动作，从而确保游戏正常进行。
    简而言之，这个判断语句用于检查环境是否有一个"FIRE"动作，以确保代理能够按需开始或继续游戏。
    '''
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)
    return env
