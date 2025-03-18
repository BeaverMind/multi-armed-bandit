# Multi-armed bandit 
Implementation of the Multi-armed bandit problem for educational purposes

# Introduction

According to [wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit): 

> The multi-armed bandit problem is a classic reinforcement learning problem that exemplifies the exploration–exploitation tradeoff dilemma. In contrast to general RL, the selected actions in bandit problems do not affect the reward distribution of the arms. The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), who has to decide which machines to play, how many times to play each machine and in which order to play them, and whether to continue with the current machine or try a different machine. The multi-armed bandit problem also falls into the broad category of stochastic scheduling.

You come across this problem everyday, whenever you are faced with choices. How do you pick the best strategy to determine, from a set of restaurants, which ones will be your favourite to go for maximized happiness? This exemplifies the exploration-exploitation tradeoff. This problem is also faced in machine learning. Additionally, multi-armed bandits have been used to model problems such as managing research projects in a large organization, like a science foundation or a pharmaceutical company (wikipedia). Here is a nice animation from Jana Beck to illustrate it ([source](https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/)). 

![Video as GIF](assets/multi_armed_bandit.gif)

# Implementation

This repo implements a simple bandit problem with the epsilon-greedy approach, Softmax/Boltzmann exploration, Upper Confidence Bound (UCB1), and Thompson Sampling.

All you need is numpy and matplotlib, then you could run:

```
python main.py
```

which should give you a file `results.pdf` that shows the expected rewards and the run time for different strategies.