"""
PTAN stands for PyTorch AgentNet -- reimplementation of AgentNet library for pytorch
"""
import setuptools

# 因为存在冲突，所以去除手动安装依赖
# requirements = ['torch==1.7.0', 'gym', 'atari-py', 'numpy', 'opencv-python']
requirements = []

setuptools.setup(
    name="ptan",
    author="Max Lapan",
    author_email="max.lapan@gmail.com",
    license='GPL-v3',
    description="PyTorch reinforcement learning framework",
    version="0.9",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
