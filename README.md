# Quixo Project

This project was a group project during a project at CentraleSupélec. This repository contains only my part of the project. The part of the report concerning my part is available in the “Rapport de projet.pdf” file and contains details on all the choices I made to achieve my goal: an AI capable of beating a human every time.

## Description

This project implements a Quixo game with various features. The code is organized in the `projet_quixo` file and allows for simulating games and testing different functionalities (see the end of the .py files for instructions on how to customize the game).

## Prerequisites

The project uses **Poetry** for dependency and version management. It is highly recommended to use Poetry to run this code in order to avoid issues related to dependencies or library versions.

### Installing Poetry

If you haven’t installed Poetry yet, you can do so by following the official instructions:
[https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

### Installing Dependencies

Once Poetry is installed, you can install the project dependencies using the following command (run it from the project directory):

```bash
poetry install
```

## Repository Overview

This repository includes different Q-learning methods to develop a strong AI capable of consistently beating a human player at Quixo (a more complex version of Connect Four).

- The `q_learning.py` file contains the first version of the game, with the classical Q-learning agent class and a `train` function to train the agents.

- The `q_learning_parallele.py` file contains the game split into two parts (graphics and logic), with the classical Q-learning agent class and a `train` function for agent training.

- The `q_learning_cano_parallele.py` file contains the game in two parts (graphics and logic), with a Q-learning agent class adapted to a canonical representation of states. It includes two `train` functions: a classic one and another that parallelizes training across CPUs.

- The `Deep_q_learning.py` file contains the game in two parts (graphics and logic), with a deep Q-learning agent class (also a linear model class from PyTorch) along with the necessary functions to train the agent.

- The `Deep_q_learning_MCTS.py` file contains the game, also split into two parts. This is the final version of the Deep Q Learning algorithm implemented with an MCTS tree search. 
