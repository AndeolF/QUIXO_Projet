import tkinter as tk
from tkinter import messagebox
import random
import pickle
from tqdm import tqdm  
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
import multiprocessing
from functools import partial
import pandas as pd
import pickletools
import os
import zlib
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple


class QuixoLogic:
    def __init__(self, N=7):
        self.N = N
        self.board = [["" for _ in range(N)] for _ in range(N)]
        self.current_player = "X"
        self.selected_position = None
        

    def is_valid_selection(self, i, j):
        return (self.est_premier_carre_interieur(i, j)) and (self.board[i][j] == "" or self.board[i][j] == self.current_player)

    def est_premier_carre_interieur(self, i, j):
        # Conditions pour être sur le premier carré intérieur
        if i == 1 and 1 <= j <= self.N - 2:
            return True
        if i == self.N - 2 and 1 <= j <= self.N - 2:
            return True
        if j == 1 and 2 <= i <= self.N - 3:
            return True
        if j == self.N - 2 and 2 <= i <= self.N - 3:
            return True
        return False

    def give_arrows_position(self, i, j):
        arrows_positions = []
        arrows_positions.append((self.N-1, j))
        arrows_positions.append((i, self.N-1))
        arrows_positions.append((i, 0))
        arrows_positions.append((0, j))
        if i == 1:
            arrows_positions.remove((0, j))
        if i == self.N-2:
            arrows_positions.remove((self.N-1, j))
        if j == 1:
            arrows_positions.remove((i, 0))
        if j == self.N-2:
            arrows_positions.remove((i, self.N-1))
        return arrows_positions

    def give_one_arrow_direction(self, arrow_position):
        x, y = arrow_position
        if x == 0:
            return "↓"
        elif x == self.N - 1:
            return "↑"
        elif y == 0:
            return "→"
        elif y == self.N - 1:
            return "←"

    def move_piece(self, arrow_direction, i, j):
        self.selected_position = (i, j)
        if arrow_direction == "↓":
            self.push_column_down(j)
        elif arrow_direction == "↑":
            self.push_column_up(j)
        elif arrow_direction == "→":
            self.push_row_right(i)
        elif arrow_direction == "←":
            self.push_row_left(i)
        

        

    def push_column_down(self, col):
        row = self.selected_position[0]
        for i in range(row, 1, -1):
            self.board[i][col] = self.board[i - 1][col]
        self.board[1][col] = self.current_player

    def push_column_up(self, col):
        row = self.selected_position[0]
        for i in range(row, self.N - 2):
            self.board[i][col] = self.board[i + 1][col]
        self.board[self.N - 2][col] = self.current_player

    def push_row_right(self, row):
        col = self.selected_position[1]
        for j in range(col, 1, -1):
            self.board[row][j] = self.board[row][j - 1]
        self.board[row][1] = self.current_player

    def push_row_left(self, row):
        col = self.selected_position[1]
        for j in range(col, self.N - 2):
            self.board[row][j] = self.board[row][j + 1]
        self.board[row][self.N - 2] = self.current_player

    def check_victory(self):
        for i in range(self.N):
            for j in range(self.N - 4):
                if all(self.board[i][j + k] == self.current_player for k in range(5)):
                    return True

        for j in range(self.N):
            for i in range(self.N - 4):
                if all(self.board[i + k][j] == self.current_player for k in range(5)):
                    return True

        for i in range(self.N - 4):
            for j in range(self.N - 4):
                if all(self.board[i + k][j + k] == self.current_player for k in range(5)):
                    return True

        for i in range(4, self.N):
            for j in range(self.N - 4):
                if all(self.board[i - k][j + k] == self.current_player for k in range(5)):
                    return True

        return False

    def is_board_full(self):
        for row in self.board:
            for cell in row:
                if cell == "":
                    return False
        return True

    def switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def reset_game(self):
        self.board = [["" for _ in range(self.N)] for _ in range(self.N)]
        self.selected_position = None
        self.current_player = "X"
        return  self.board


    def get_available_actions(self, state):
        # Retourne une liste des actions possibles pour l'IA
        actions = []
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                if self.is_valid_selection(i, j):
                    arrows_postions = self.give_arrows_position(i, j)
                    for arrow_postions in arrows_postions:
                        direction =  self.give_one_arrow_direction(arrow_postions)
                        actions.append((i, j, direction))
        return actions
    

    def step(self, action):
        i, j, arrow_direction = action
        done_type = "rien"
        self.move_piece(arrow_direction,i,j)
        if self.check_victory():
            done_type = "Victoire"
        if self.is_board_full():
            done_type = "NUL"
        self.switch_player()
        done = False
        next_state = self.board  # L'état après l'action
        reward = self.compute_reward(done_type,self.current_player)  # Calculer la récompense
        if done_type != "rien":
            done = True
        return next_state, reward, done
    
    def compute_reward(self, done_type, type_player):

        if done_type == "Victoire":
            return 50
        if done_type == "NUL": 
            return 25
        else:
            for i in range(self.N):

                # Vérification des lignes
                if self.board[i].count(type_player) == 2 and self.board[i].count("") == self.N - 2:
                    return 1
                if self.board[i].count(type_player) == 3 and self.board[i].count("") == self.N - 3:
                    return 5
                if self.board[i].count("X") == 4 and self.board[i].count("") == self.N - 4:
                    return 10
            
                # Vérification des colonnes
                column = [self.board[j][i] for j in range(self.N)]
                if column.count(type_player) == 2 and column.count("") == self.N - 2:
                    return 1
                if column.count(type_player) == 3 and column.count("") == self.N - 3:
                    return 5
                if column.count(type_player) == 4 and column.count("") == self.N - 4:
                    return 10

            # Vérification des diagonales
            # Diagonale principale
            diag1 = [self.board[i][i] for i in range(self.N)]
            if diag1.count(type_player) == 2 and diag1.count("") == self.N - 2:
                return 1
            if diag1.count(type_player) == 3 and diag1.count("") == self.N - 3:
                return 5
            if diag1.count(type_player) == 4 and diag1.count("") == self.N - 4:
                return 10

            # Diagonale secondaire
            diag2 = [self.board[i][self.N - 1 - i] for i in range(self.N)]
            if diag2.count(type_player) == 2 and diag2.count("") == self.N - 2:
                return 1
            if diag2.count(type_player) == 3 and diag2.count("") == self.N - 3:
                return 5
            if diag2.count(type_player) == 4 and diag2.count("") == self.N - 4:
                return 10

            # Si aucune condition n'est remplie, retourner -1
            return -1
    
    




class QuixoGame:
    def __init__(self, root, agent = None):
        self.root = root
        self.root.title("Jeu Quixo")
        self.agent = agent
        self.logic = QuixoLogic()  # Utilisation de la classe logique
        self.selected_position = None

        self.canvas = tk.Canvas(root, width=900, height=900)
        self.canvas.pack()

        self.status_label = tk.Label(root, text=f"Joueur courant : {self.logic.current_player}", font=('Arial', 18))
        self.status_label.pack(side=tk.RIGHT, padx=20)

        self.buttons = [[None for _ in range(self.logic.N)] for _ in range(self.logic.N)]
        for i in range(self.logic.N):
            for j in range(self.logic.N):
                btn = tk.Button(root, text="", font=('Arial', 24), width=4, height=2,
                                command=lambda i=i, j=j: self.select_square(i, j))
                btn_window = self.canvas.create_window(100 + j * 80, 100 + i * 80, window=btn)
                self.buttons[i][j] = btn

        self.arrow_buttons = []

        if self.agent and self.logic.current_player == "O":
            self.root.after(500, self.ai_play)

    def select_square(self, i, j):
        if self.logic.is_valid_selection(i, j):
            self.clear_selection()
            self.selected_position = (i, j)
            self.buttons[i][j].config(bg='gray')
            self.show_arrows(i, j)

    def show_arrows(self, i, j):
        arrows_positions = self.logic.give_arrows_position(i, j)
        for arrow_position in arrows_positions:
            x, y = arrow_position
            arrow_direction = self.logic.give_one_arrow_direction(arrow_position)
            self.buttons[x][y].config(text=arrow_direction, command=lambda x=x, y=y: self.select_arrow(x, y))
            self.arrow_buttons.append((x, y))

    def select_arrow(self, x, y):
        arrow_direction = self.buttons[x][y].cget("text")
        if arrow_direction in ["↓", "↑", "→", "←"]:
            i, j = self.selected_position
            self.logic.move_piece(arrow_direction, i, j)
            self.clear_selection()
            self.update_board()
            self.stoper_si_fin()
            self.logic.switch_player()


    def stoper_si_fin(self):
        if self.logic.check_victory():
            print("victoire")
            messagebox.showinfo("Victoire", f"Le joueur {self.logic.current_player} a gagné!")
            self.reset_game()
        elif self.logic.is_board_full():
            messagebox.showinfo("Match nul !")
            self.reset_game()
        else:
            self.update_status()
            if self.agent and self.logic.current_player == "O":
                self.root.after(500, self.ai_play)

    def update_board(self):
        for i in range(self.logic.N):
            for j in range(self.logic.N):
                self.buttons[i][j].config(text=self.logic.board[i][j])
        self.clear_arrows()

    def update_status(self):
        self.status_label.config(text=f"Joueur courant : {self.logic.current_player}")

    def clear_selection(self):
        if self.selected_position:
            i, j = self.selected_position
            self.buttons[i][j].config(bg='light gray')
            self.selected_position = None
        self.clear_arrows()

    def clear_arrows(self):
        for (x, y) in self.arrow_buttons:
            self.buttons[x][y].config(text="")
        self.arrow_buttons = []

    def reset_game(self):
        board = self.logic.reset_game()
        self.update_board()
        self.update_status()
        return board  # Retourner l'état initial du jeu

    def ai_play(self):
        state = self.logic.board  # Obtient l'état actuel du jeu
        available_actions = self.get_available_actions(state)
        action = self.agent.choose_action(state, available_actions)  # Utilise la Q-table chargée pour choisir la meilleure action
        next_state, reward, done = self.step(action)
        return next_state, reward, done
    



# class QLearningAgent:
#     compteur = 0
#     def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2,type_player="O"):
#         self.q_table = {}  # Dictionnaire pour stocker les valeurs Q
#         self.alpha = alpha  # Taux d'apprentissage
#         self.gamma = gamma  # Facteur de discount
#         self.epsilon = epsilon  # Paramètre d'exploration
#         self.type_player = type_player # Joueur X ou O
#         self.canonical_cache = {} # Cache pour les représentations canoniques

#     def get_q_value(self, state, action):
#         """Retourne la valeur Q pour un état et une action donnés."""
#         state_tuple = tuple(tuple(row) for row in state)
#         return self.q_table.get((state_tuple, action), 0.0)



#     def choose_action(self, state, available_actions):
#         """Choisit une action en fonction de la stratégie d'exploration/exploitation."""
#         if random.uniform(0, 1) < self.epsilon:
#             return random.choice(available_actions)  # Exploration
#         else:
#             # Calculer les Q-values canoniques pour toutes les actions disponibles
#             q_values = [self.get_q_value(state, a) for a in available_actions]
#             max_q_value = max(q_values)
#             max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q_value]

#             chosen_action = random.choice(max_actions)  # Exploitation

#             return chosen_action

#     def update_q_value(self, game, state, action, reward, next_state):
#         """Met à jour la valeur Q pour un état et une action donnés."""
#         # Convertir chaque sous-liste dans 'state' et 'next_state' en un tuple
#         state_tuple = tuple(tuple(row) for row in state)
#         next_state_tuple = tuple(tuple(row) for row in next_state)

#         next_available_actions = game.get_available_actions(next_state)
#         max_q_next = max([self.get_q_value(next_state_tuple, a) for a in next_available_actions], default=0.0)

#         # Récupérer la Q-value pour l'état et l'action actuels sous forme canonique
#         old_q_value = self.get_q_value(state_tuple, action)
#         # print(old_q_value)

#         # Calculer la nouvelle Q-value
#         new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_q_next - old_q_value)

#         self.q_table[(state_tuple, action)] = new_q_value




#     def train(self, game, episodes=1000, rival=None):
#         """Entraîne l'agent en jouant plusieurs épisodes."""
#         rewards_per_episode_perso = []
#         entropy_per_episode_perso = []

#         rewards_per_episode_rival = []
#         entropy_per_episode_rival = []

#         for episode in tqdm(range(episodes), desc="Training Progress"):
#             state = game.reset_game()  # Réinitialiser l'état du jeu
#             done = False
#             rival_done = False

#             episode_reward_perso = 0
#             episode_reward_rival= 0

#             tab_state = [state]
#             tab_rival_state = []



#             action_distribution_perso = defaultdict(int)
            
#             action_distribution_rival = defaultdict(int)

#             while not done :





#                 available_actions = game.get_available_actions(state)
#                 action = self.choose_action(state, available_actions)

#                 action_distribution_perso[action] += 1

#                 rival_state, reward, done = game.step(action)



                
#                 # le rival joue 
#                 if done == False:
#                     available_actions = game.get_available_actions(rival_state)
#                     rival_action = rival.choose_action(state, available_actions)

#                     action_distribution_rival[rival_action] += 1
                
#                     next_state, rival_reward, rival_done = game.step(rival_action)

#                     # df_1 = pd.DataFrame(state)
#                     # print(df_1)

#                     # df_2 = pd.DataFrame(rival_state)
#                     # print(df_2)                    


#                     rival.update_q_value(game,rival_state, rival_action, rival_reward, next_state)

#                 if done == True:
#                     rival_reward = -50
#                     # print(tab_rival_state)
#                     rival.update_q_value(game,tab_rival_state[-1], rival_action, rival_reward, next_state)
                
#                 tab_rival_state.append(rival_state)

#                 if rival_done == True:
#                     reward = -50
#                     self.update_q_value(game,state, action, reward, rival_state)
#                     episode_reward_perso += reward
#                     episode_reward_rival += rival_reward
#                     break
#                 else:
#                     self.update_q_value(game,state, action, reward, rival_state)
#                 state = next_state
#                 tab_state.append(state)




#                 # episode_reward_perso += reward
#                 # episode_reward_rival += rival_reward


#             # print(self.q_table)
#         #     # Calculer l'entropie pour cet épisode
#         #     total_actions_perso = sum(action_distribution_perso.values())
#         #     action_probabilities_perso = np.array([count / total_actions_perso for count in action_distribution_perso.values()])

#         #     episode_entropy_perso = -np.sum(action_probabilities_perso * np.log(action_probabilities_perso + 1e-10))  # Entropie de Shannon

#         #     total_actions_rival = sum(action_distribution_rival.values())
#         #     action_probabilities_rival = np.array([count / total_actions_rival for count in action_distribution_rival.values()])

#         #     episode_entropy_rival = -np.sum(action_probabilities_rival * np.log(action_probabilities_rival + 1e-10))  # Entropie de Shannon



#         # # # Stocker la récompense et l'entropie de cet épisode
#         #     rewards_per_episode_perso.append(episode_reward_perso)
#         #     entropy_per_episode_perso.append(episode_entropy_perso)

#         #     rewards_per_episode_rival.append(episode_reward_rival)
#         #     entropy_per_episode_rival.append(episode_entropy_rival)

#         # print(f"Rewards_per_episode_perso: {rewards_per_episode_perso}\n",f"Rewards_per_episode_rival: {rewards_per_episode_rival}\n" )

#         # self.plot_training_results(rewards_per_episode_perso, entropy_per_episode_perso)
#         # self.plot_training_results(rewards_per_episode_rival, entropy_per_episode_rival)

#         return self.q_table, rival.q_table

#     def plot_training_results(self, rewards, entropy):
#         episodes = np.arange(len(rewards))

#         # Graphique des récompenses
#         plt.figure(figsize=(12, 5))

#         plt.subplot(1, 2, 1)
#         plt.plot(episodes, rewards, label="Reward")
#         plt.xlabel("Episodes")
#         plt.ylabel("Total Reward")
#         plt.title("Reward per Episode")
#         plt.grid(True)
#         plt.legend()

#         # Graphique de l'entropie
#         plt.subplot(1, 2, 2)
#         plt.plot(episodes, entropy, label="Entropy", color='orange')
#         plt.xlabel("Episodes")
#         plt.ylabel("Entropy")
#         plt.title("Entropy of Actions per Episode")
#         plt.grid(True)
#         plt.legend()

#         plt.tight_layout()
#         plt.show()


#     def afficher_q_table(self):
#         print(len(self.q_table))
#         return 0






#     def load_q_table(self, filename):
#         # Charge la Q-table depuis un fichier
#         # self.q_table = joblib.load(path)
#         with open(filename, 'rb') as f:
#             self.q_table = pickle.load(f)




# def save_q_table(q_table, path):
#     temp_filename = path + '.tmp'
#     joblib.dump(q_table, temp_filename, compress=3)
#     os.replace(temp_filename, path)




# Définition de la classe DQN (réseau principal et cible)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Expérience Replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    


# Initialisation des paramètres et du réseau
valid_indices = [1, 0, 9, 8, 10, 13, 12, 14, 17, 16, 18, 5, 6, 33, 32, 35, 45, 46, 47, 37, 36, 39, 49, 50, 51, 41, 40, 43, 53, 54, 55, 4, 7, 20, 22, 23, 24, 26, 27, 28, 30, 31, 2, 3]
output_dim = len(valid_indices)  # 44
index_to_valid_index = {i: valid_index for i, valid_index in enumerate(valid_indices)}
valid_index_to_index = {valid_index: i for i, valid_index in enumerate(valid_indices)}



input_dim = 5 * 5 * 3  # 5x5 plateau et 3 canaux pour "", "X", "O"
batch_size = 64
gamma = 0.99  # Facteur d'actualisation
epsilon = 0.00001  # Taux d'exploration initial
learning_rate = 0.001


policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(10000)

# Sélection de l'action avec une stratégie epsilon-greedy
steps_done = 0



def action_to_index(i, j, direction):
    direction_map = {"←": 0, "↑": 1, "→": 2, "↓": 3}
    
    if (i, j) == (1, 1):  # Coin en haut à gauche
        if direction in ["←", "↑"]:
            return direction_map[direction]
        
    elif (i, j) == (5, 5):  # Coin en bas à droite
        if direction in ["→", "↓"]:
            return direction_map[direction]
    
    elif (i, j) == (1, 5):  # Coin en haut à droite
        if direction in ["→", "↑"]:
            return 4 + direction_map[direction]
    
    elif (i, j) == (5, 1):  # Coin en bas à gauche
        if direction in ["←", "↓"]:
            return 4 + direction_map[direction]
    
    else:
        # Pour les autres positions sur les bords
        base_index = 8  # On saute les 8 premières actions des coins
        
        if i == 1:  # Bord supérieur
            return base_index + (j - 2) * 4 + direction_map[direction]
        elif i == 5:  # Bord inférieur
            return base_index + 12 + (j - 2) * 4 + direction_map[direction]
        elif j == 1:  # Bord gauche
            return base_index + 24 + (i - 2) * 4 + direction_map[direction]
        elif j == 5:  # Bord droit
            return base_index + 36 + (i - 2) * 4 + direction_map[direction]
        
    raise ValueError("Invalid action")


def index_to_action(index):
    direction_map = ["←", "↑", "→", "↓"]

    # Coins
    if index < 2:
        return (1, 1, direction_map[index])  # Coin en haut à gauche
    if index == 2:
        return (5, 5, '→')
    if index == 3:
        return (5, 5, '↓')
    elif index < 6:
        return (1, 5, direction_map[index - 4])  # Coin en haut à droite
    if index == 6:
        return (1, 5, '→')
    if index == 7:
        return (5, 1, '↓')

    # Bords
    else:
        base_index = index - 8

        if base_index < 12:  # Bord supérieur (i = 1)
            j = (base_index // 4) + 2
            direction = direction_map[base_index % 4]
            return (1, j, direction)
        
        elif base_index < 24:  # Bord inférieur (i = 5)
            j = ((base_index - 12) // 4) + 2
            direction = direction_map[(base_index - 12) % 4]
            return (5, j, direction)
        
        elif base_index < 36:  # Bord gauche (j = 1)
            i = ((base_index - 24) // 4) + 2
            direction = direction_map[(base_index - 24) % 4]
            return (i, 1, direction)
        
        elif base_index < 48:  # Bord droit (j = 5)
            i = ((base_index - 36) // 4) + 2
            direction = direction_map[(base_index - 36) % 4]
            return (i, 5, direction)

    raise ValueError("Invalid index")


def extract_state_from_board(board):
    # Extraire l'état 5x5 du tableau 7x7 self.board
    state = [row[1:6] for row in board[1:6]]
    return state




def get_available_actions_indices(game,state):
    available_actions = game.get_available_actions(state)  # Renvoie une liste de tuples (i, j, direction)
    return [action_to_index(i, j, direction) for (i, j, direction) in available_actions]





def select_action(state):
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # Le réseau prédit un index parmi les 44
            index_pred = policy_net(state).argmax(dim=1).item()
            # Convertir cela en l'indice d'action correspondant
            action_index = index_to_valid_index[index_pred]
            return action_index
    else:
        # Choisir aléatoirement parmi les actions disponibles
        action_index = random.choice(valid_indices)
        return action_index






def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # Convertir les actions en indices dans la sortie du réseau
    action_index_batch = torch.tensor([valid_index_to_index[action.item()] for action in action_batch])

    # Q(s, a) pour les actions prises
    state_action_values = policy_net(state_batch).gather(1, action_index_batch.unsqueeze(1))

    # Double DQN : choisir les meilleures actions avec le policy_net, évaluer avec le target_net
    next_state_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
    next_state_values = target_net(next_state_batch).gather(1, next_state_actions).detach()

    expected_state_action_values = reward_batch + (gamma * next_state_values)

    # Calcul de la perte (Huber loss)
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

    # Optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# encodage one-hot

def encode_state(state):
    encoded_state = []
    for row in state:
        for cell in row:
            if cell == "":
                encoded_state.extend([1, 0, 0])
            elif cell == "X":
                encoded_state.extend([0, 1, 0])
            elif cell == "O":
                encoded_state.extend([0, 0, 1])
    return encoded_state



    # Mise à jour périodique du réseau cible
def update_target_net():
    target_net.load_state_dict(policy_net.state_dict())




def train_DQL(game, num_episodes):
    valid_indices = [1, 0, 9, 8, 10, 13, 12, 14, 17, 16, 18, 5, 6, 33, 32, 35, 45, 46, 47, 37, 36, 39, 49, 50, 51, 41, 40, 43, 53, 54, 55, 4, 7, 20, 22, 23, 24, 26, 27, 28, 30, 31, 2, 3]
    index_to_valid_index = {i: valid_index for i, valid_index in enumerate(valid_indices)}
    valid_index_to_index = {valid_index: i for i, valid_index in enumerate(valid_indices)}

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state = extract_state_from_board(game.reset_game())  # Réinitialiser l'état du jeu
        state_tensor = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)
    
        for t in range(200):  # Limite le nombre d'étapes par épisode
            print(t)
            available_actions_indices = get_available_actions_indices(game, state)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    # Obtenir les valeurs Q pour les indices valides
                    q_values = policy_net(state_tensor)
                    valid_q_values = q_values[0, [valid_index_to_index[i] for i in available_actions_indices]]
                    
                    # Sélectionner l'indice de l'action avec la plus haute valeur Q
                    action_index = available_actions_indices[valid_q_values.argmax().item()]
            else:
                action_index = random.choice(available_actions_indices)
        
            # Décoder l'indice en une action (i, j, direction)
            action = index_to_action(action_index)
        
            # Exécuter l'action
            next_board, reward, done = game.step(action)
            next_state = extract_state_from_board(next_board)
            next_state_tensor = torch.tensor(encode_state(next_state), dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
            # Stocker la transition en mémoire
            memory.push(state_tensor, torch.tensor([[valid_index_to_index[action_index]]], dtype=torch.long), next_state_tensor, reward_tensor)
        
            # Passer à l'état suivant
            state_tensor = next_state_tensor
        
            # Optimiser le modèle
            optimize_model()
        
            if done:
                break
    
        # Mettre à jour le réseau cible périodiquement
        if episode % 10 == 0:
            update_target_net()






game = QuixoLogic(N=7)
train_DQL(game,1)