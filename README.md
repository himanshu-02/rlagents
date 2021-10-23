# rlagents
## Understanding Reinforcement Learning and Implementing RL agents for OpenAI Gym from Scratch

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [rlagents](#rlagents)
  - [Understanding Reinforcement Learning and Implementing RL agents for OpenAI Gym from Scratch](#understanding-reinforcement-learning-and-implementing-rl-agents-for-openai-gym-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
- [Reinforcement Learning](#reinforcement-learning)
    - [Tech Stack](#tech-stack)
    - [File Structure](#file-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Results and Demo](#results-and-demo)
  - [Future Work](#future-work)
  - [Troubleshooting](#troubleshooting)
  - [Contributors](#contributors)
  - [Acknowledgements and Resources](#acknowledgements-and-resources)
  - [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project

# Reinforcement Learning 
![cool icon](https://i.imgur.com/bUeBdxC.png)

Aim and Description of project.  
Refer this [documentation](https://link/to/report/)

### Tech Stack
These are some of the technologies we used in this project. 
- [OpenAI gym](https://gym.openai.com/)
- [Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [Jupyter Notebook](https://jupyter.org/)  

### File Structure
    .
    ├── app.py                  # Explain the function preformed by this file in short
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── report.pdf          # Project report
    │   └── notes               # Folder containing markdown notes of lectures 
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    ├── ...
    ├── tools                   # Tools and utilities
    ├── LICENSE
    ├── README.md 
    ├── Setup.md                # If Installation instructions are lengthy
    └── todo.md                 # If Future developments and current status gets lengthy
    

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
How to run the driver code
```


<!-- RESULTS AND DEMO -->
## Results and Demo
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.  
[**result screenshots**](https://result.png)  
![**result gif or video**](https://result.gif)  

| Use  |  Table  |
|:----:|:-------:| 
| For  | Comparison|


<!-- FUTURE WORK -->
## Future Work
* See [todo.md](https://todo.md) for seeing developments of this project
- [ ] Making a custom environment 
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4


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
* [SRA VJTI](http://sra.vjti.info/) | Eklavya 2021  
* Refered [this](https://link) for achieving this  
...


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project. 
