# rlagents
## Understanding Reinforcement Learning and Implementing RL agents for OpenAI Gym from Scratch

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [rlagents](#rlagents)
  - [Understanding Reinforcement Learning and Implementing RL agents for OpenAI Gym from Scratch](#understanding-reinforcement-learning-and-implementing-rl-agents-for-openai-gym-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
- [Reinforcement Learning](#reinforcement-learning)
  - [Aim:](#aim)
  - [Theory:](#theory)
    - [Reinforcement Learning](#reinforcement-learning-1)
    - [Markov Decision Process](#markov-decision-process)
    - [Multi armed Bandit Problem](#multi-armed-bandit-problem)
    - [Epsilon Greedy Algorithm](#epsilon-greedy-algorithm)
    - [OpenAI gym](#openai-gym)
    - [Agents used](#agents-used)
    - [Algorithms used](#algorithms-used)
    - [Tech Stack](#tech-stack)
    - [File Structure](#file-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [## Results and Demo](#-results-and-demo)
    - [CartPole](#cartpole)
    - [Mountain Car](#mountain-car)
    - [Lunar Lander](#lunar-lander)
    - [Taxi](#taxi)
    - [Car racing](#car-racing)
  - [Future Work](#future-work)
  - [Troubleshooting](#troubleshooting)
  - [Contributors](#contributors)
  - [Acknowledgements and Resources](#acknowledgements-and-resources)
  - [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project

# Reinforcement Learning 
![cool icon](https://i.imgur.com/bUeBdxC.png)

## Aim:
- Understand Reinforcement Learning
- A simple solution to the Multi Armed Bandit Problem
- To solve rl agents in OpenAI gym

## Theory: 
Refer our [documentation](https://docs.google.com/document/d/1DccC34vfk1-0BPo3-_cWzmp5WbfnomgBwjTl0GaGUIU/edit?usp=sharing) for detailed analysis and brief overview of our project. 

---
 
### Reinforcement Learning 
- Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones. 
- In general, a reinforcement learning agent is able to perceive and interpret its environment, take actions and learn through trial and error.
-Some notable examples of RL in particular Deep RL
- In 2013, Atari game Breakout took around 36 hours of training with the DQN in order to achieve commendable results! Now we can achieve similar results in matter of hours.
- The agents created in Dota2 were able to defeat pro players at their own game! And did really well in the 5v5 matchup!! 
- As you can see DeepMind by Google and OpenAI are two organisations with insane     accomplishments in the field of Reinforcement Learning

### Markov Decision Process
![image](https://miro.medium.com/max/700/1*ywOrdJAHgSL5RP-AuxsfJQ.png)
- The learner and decision maker is called the agent. 
- The thing it interacts with, comprising everything outside the agent, is called the environment.
- These interact continually, the agent selecting actions and the environment responding to these actions and presenting new situations to the agent.
- The environment also gives rise to rewards, special numerical values that the agent seeks to maximize over time through its choice of actions.
- Basically, If you have a problem you want to solve, if you can map it to an MDP, it means you can run a reinforcement algorithm on it

### Multi armed Bandit Problem
- The multi-armed bandit problem is a classic problem that well demonstrates the exploration vs exploitation dilemma. Imagine you are in a casino facing multiple slot machines and each is configured with an unknown probability of how likely you can get a reward at one play. The question is: What is the best strategy to achieve highest long-term rewards?
- We are using Epsilon greedy Algorithm to solve this problem

### Epsilon Greedy Algorithm
- The Epsilon-Greedy algorithm balances exploitation and exploration fairly basically. 
- It takes a parameter, epsilon, between 0 and 1, as the probability of exploring the options (called arms in multi-armed bandit discussions) as opposed to exploiting the current best variant in the test. 
- For example, say epsilon is set at 0.1. 
- Every time a visitor comes to the website being tested, a number between 0 and 1 is randomly drawn. If that number is greater than 0.1, then that visitor will be shown whichever variant (at first, version A) is performing best. 
- If that random number is less than 0.1, then a random arm out of all available options will be chosen and provided to the visitor. 
- The visitor???s reaction will be recorded (a click or no click, a win or lose, etc.) and the success rate of that arm will be updated accordingly. 
Low values of epsilon correspond to less exploration and more exploitation, therefore - it takes the algorithm longer to discover which is the best arm but once found, it exploits it at a higher rate. 

### OpenAI gym
- Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Pinball.
  
### Agents used
- CartPole
- Mountain-Car
- LunarLander
- Self Driving Racing-Car
- Taxi Driver 

### Algorithms used
- Epsilon Greedy
- PPO
- Q-Learning
- DQN 

### Tech Stack
These are some of the technologies we used in this project. 
- [OpenAI gym](https://gym.openai.com/)
- [Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [Jupyter Notebook](https://jupyter.org/)  

### File Structure
    .
    ????????? app.py                  # Explain the function preformed by this file in short
    ????????? docs                    # Documentation files (alternatively `doc`)
    ???   ????????? report.pdf          # Project report
    ???   ????????? notes               # Folder containing markdown notes of lectures 
    ????????? src                     # Source files (alternatively `lib` or `app`)
    ???   ????????? Training/saved      # Trained Model of CartPole, CarRacing
    ???   ????????? Agent Codes         # All the agent codes
    ????????? LICENSE
    ????????? README.md 
 
    

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- OpenAI gym 
  - You can visit the [OpenAI gym Repo](https://github.com/openai/gym) or their [documentation](https://gym.openai.com/docs/) for the installation steps.

- Stable-baselines3 
  - You can visit the installation section of Stable-baselines3 docs [here](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

- Jupyter-notebook
  - refer [here](https://jupyter.org/install)
  
* For OpenAI gym 
```sh
pip install gym

pip install gym[atari]    #For all atari dependencies

pip install gym[all]    #For all dependencies
```
- For Stable-baselines3
```sh
pip install stable-baselines3

pip install stable-baselines3[extra]    #use this if you want dependencies like Tensorboard, OpenCV, Atari-py
```
- Note: Some shells such as Zsh require quotation marks around brackets, i.e.
  
```pip install 'gym[all]' ```

### Installation
1. Clone the repo
```sh
git clone https://github.com/himanshu-02/rlagents
```


<!-- USAGE EXAMPLES -->
## Usage
```
Clone the environment.
Use our codes on jupyter notebook.
You can use our saved models as well. 
```


<!-- RESULTS AND DEMO -->
## Results and Demo
---
### CartPole
- Maximum reward of 200 is achieved by the agent

![Cartpole](docs/videos/solved_cartpole.gif)

### Mountain Car
- First clear before 200 episodes every time 

![mountian](docs/videos/solved_mountaincar.gif)

### Lunar Lander
- Solved using DQN and after training good results are achieved

![Lunar](docs/videos/solved_lunar_lander.gif)

### Taxi 
- Solved using Q learning and done perfectly

![taxi](docs/videos/solved_taxicar.gif)

### Car racing
- Solved using PPO after training for 2m steps, higest score around 700/900 is achieved

![car](docs/videos/solved_racing_car.gif)


<!-- FUTURE WORK -->
## Future Work
* See [todo.md](https://todo.md) for seeing developments of this project
- [ ] Creating a custom environment 
- [ ] Coding Agents using various Models such as Deep Q-Learning, PPO, etc to train the custom environment and compare the results.
- [ ] Completing more advanced environments available on OpenAI gym 


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Make sure you are using the correct environment name 
* Incase you missed it, Note: Some shells such as Zsh require quotation marks around brackets, i.e.
  
```pip install 'gym[all]' ```


<!-- CONTRIBUTORS -->
## Contributors
* [Chirag Shelar](https://github.com/Locoya)
* [Himanshu Chougule](https://github.com/himanshu-02)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](https://sravjti.in/) | Eklavya 2021  
...


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project. 
