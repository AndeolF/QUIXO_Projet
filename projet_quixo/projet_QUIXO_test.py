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
        return self.board


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
        reward = self.compute_reward(done_type,self.current_player)  # Calculer la récompense
        self.switch_player()
        done = False
        next_state = self.board  # L'état après l'action
        
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
        self.logic.reset_game()
        self.update_board()
        self.update_status()

    def ai_play(self):
        state = self.board  # Obtient l'état actuel du jeu
        available_actions = self.get_available_actions(state)
        action = self.agent.choose_action(state, available_actions)  # Utilise la Q-table chargée pour choisir la meilleure action
        next_state, reward, done = self.step(action)
        return next_state, reward, done
    



class QLearningAgent:
    compteur = 0
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2,type_player="O"):
        self.q_table = {}  # Dictionnaire pour stocker les valeurs Q
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Paramètre d'exploration
        self.type_player = type_player # Joueur X ou O
        self.canonical_cache = {} # Cache pour les représentations canoniques

    def get_q_value(self, state, action):
        """Retourne la valeur Q pour un état et une action donnés."""
        canonical_state, canonical_action = self.canonical_state_action((state, action))
        return self.q_table.get((canonical_state, canonical_action), 0.0)

    def choose_action(self, state, available_actions):
        """Choisit une action en fonction de la stratégie d'exploration/exploitation."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)  # Exploration
        else:
            # Calculer les Q-values canoniques pour toutes les actions disponibles
            q_values = [self.get_q_value(state, a) for a in available_actions]
            max_q_value = max(q_values)
            max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q_value]

            chosen_action = random.choice(max_actions)  # Exploitation

            # Trouver l'action originale correspondant à l'action choisie dans l'état courant
            _, canonical_action = self.canonical_state_action((state, chosen_action))
            for a in available_actions:
                _, original_canonical_action = self.canonical_state_action((state, a))
                if original_canonical_action == canonical_action:
                    return a

    def update_q_value(self, game, state, action, reward, next_state):
        """Met à jour la valeur Q pour un état et une action donnés."""
        next_available_actions = game.get_available_actions(next_state)
        max_q_next = max([self.get_q_value(next_state, a) for a in next_available_actions], default=0.0)

        # Récupérer la Q-value pour l'état et l'action actuels sous forme canonique
        old_q_value = self.get_q_value(state, action)
        # print(old_q_value)

        # Calculer la nouvelle Q-value
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_q_next - old_q_value)

        # Stocker la nouvelle Q-value sous la forme canonique
        canonical_state, canonical_action = self.canonical_state_action((state, action))
        self.q_table[(canonical_state, canonical_action)] = new_q_value

    @staticmethod
    def rotate_direction(direction, rotation):
        """Ajuste la direction après une rotation du plateau."""
        directions = ['↑', '→', '↓', '←']
        idx = directions.index(direction)
        new_idx = (idx + rotation) % 4
        return directions[new_idx]

    @staticmethod
    def rotate_state_action(state, action, rotation):
        """Applique une rotation de 90 degrés au plateau de jeu, ajuste l'action (coordonnées et direction)."""
        # Rotation du plateau de jeu
        state_rot = np.rot90(np.array(state), k=rotation)
    
        # Ajustement de l'action
        row, col, direction = action
        for _ in range(rotation):
            row, col = col, len(state) - 1 - row
    
        direction = QLearningAgent.rotate_direction(direction, rotation)
    
        return tuple(map(tuple, state_rot)), (row, col, direction)

    @staticmethod
    def reflect_state_action(state, action, axis):
        """Applique une réflexion au plateau de jeu et ajuste l'action (coordonnées et direction)."""
        state_reflected = np.array(state)
        row, col, direction = action

        if axis == 'horizontal':
            state_reflected = np.flipud(state_reflected)
            row = len(state) - 1 - row
            if direction == '↑':
                direction = '↓'
            elif direction == '↓':
                direction = '↑'
        elif axis == 'vertical':
            state_reflected = np.fliplr(state_reflected)
            col = len(state[0]) - 1 - col
            if direction == '→':
                direction = '←'
            elif direction == '←':
                direction = '→'
        elif axis == 'main_diagonal':
            state_reflected = np.transpose(state_reflected)
            row, col = col, row
            direction = QLearningAgent.rotate_direction(direction, 1)
        elif axis == 'anti_diagonal':
            state_reflected = np.flipud(np.fliplr(np.transpose(state_reflected)))
            row, col = len(state) - 1 - col, len(state[0]) - 1 - row
            direction = QLearningAgent.rotate_direction(direction, 3)

        return tuple(map(tuple, state_reflected)), (row, col, direction)

    @staticmethod
    def invert_symbols(state):
        """Inverse les symboles 'X' et 'O' dans l'état du jeu."""
        state_inverted = np.array(state)
        state_inverted[state_inverted == 'X'] = 'O'
        state_inverted[state_inverted == 'O'] = 'X'
        return tuple(map(tuple, state_inverted))


    def canonical_state_action(self,q_value):
        """Retourne la représentation canonique d'une Q-value en tenant compte des rotations et des réflexions."""

        
        state, action = q_value
        # Convertir l'état et l'action en tuples (immuables) pour l'utiliser comme clé
        state = tuple(map(tuple, state))
        action = tuple(action)

        # Vérifier si la représentation canonique est déjà dans le cache
        if (state, action) in self.canonical_cache:
            return self.canonical_cache[(state, action)]

        # Générer toutes les rotations de l'état et de l'action
        transformations = []
        for k in range(4):
            rotated_state, rotated_action = QLearningAgent.rotate_state_action(state, action, k)
            transformations.append((rotated_state, rotated_action))

        # Ajouter les réflexions horizontale, verticale, diagonale principale, diagonale secondaire
        axes = ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal']
        for axis in axes:
            reflected_state, reflected_action = QLearningAgent.reflect_state_action(state, action, axis)
            transformations.append((reflected_state, reflected_action))
            # for k in range(4):
            #     rotated_reflected_state, rotated_reflected_action = QLearningAgent.rotate_state_action(reflected_state, reflected_action, k)
            #     transformations.append((rotated_reflected_state, rotated_reflected_action))

        # Choisir la plus petite représentation canonique
        canonical_representation = min(transformations)
        # Stocker le résultat dans le cache
        self.canonical_cache[(state, action)] = canonical_representation
        return canonical_representation

    def train(self, game, episodes=1000, rival=None):
        """Entraîne l'agent en jouant plusieurs épisodes."""
        rewards_per_episode_perso = []
        entropy_per_episode_perso = []

        rewards_per_episode_rival = []
        entropy_per_episode_rival = []

        for episode in tqdm(range(episodes), desc="Training Progress"):
        # for episode in range(episodes):
        #     print(f"episode : {episode}")
            state = game.reset_game()  # Réinitialiser l'état du jeu
            done = False
            rival_done = False

            episode_reward_perso = 0
            episode_reward_rival= 0

            tab_state = [state]
            tab_rival_state = []


            action_distribution_perso = defaultdict(int)
            
            action_distribution_rival = defaultdict(int)

            while not done :




                available_actions = game.get_available_actions(state)
                action = self.choose_action(state, available_actions)

                action_distribution_perso[action] += 1

                rival_state, reward, done = game.step(action)

                
                # le rival joue 
                if done == False:
                    available_actions = game.get_available_actions(rival_state)
                    rival_action = rival.choose_action(state, available_actions)

                    action_distribution_rival[rival_action] += 1
                
                    next_state, rival_reward, rival_done = game.step(rival_action)
                    rival.update_q_value(game,state, rival_action, rival_reward, next_state)

                if done == True:
                    rival_reward = -50
                    rival.update_q_value(game,tab_rival_state[-1], rival_action, rival_reward, next_state)
                
                tab_rival_state.append(rival_state)

                if rival_done == True:
                    reward = -50
                    self.update_q_value(game,state, action, reward, rival_state)
                    episode_reward_perso += reward
                    episode_reward_rival += rival_reward
                    break
                else:
                    self.update_q_value(game,state, action, reward, rival_state)
                state = next_state
                tab_state.append(state)




            #     episode_reward_perso += reward
            #     episode_reward_rival += rival_reward


            # # print(self.q_table)
            # # Calculer l'entropie pour cet épisode
            # total_actions_perso = sum(action_distribution_perso.values())
            # action_probabilities_perso = np.array([count / total_actions_perso for count in action_distribution_perso.values()])

            # episode_entropy_perso = -np.sum(action_probabilities_perso * np.log(action_probabilities_perso + 1e-10))  # Entropie de Shannon

            # total_actions_rival = sum(action_distribution_rival.values())
            # action_probabilities_rival = np.array([count / total_actions_rival for count in action_distribution_rival.values()])

            # episode_entropy_rival = -np.sum(action_probabilities_rival * np.log(action_probabilities_rival + 1e-10))  # Entropie de Shannon



        # # Stocker la récompense et l'entropie de cet épisode
            # rewards_per_episode_perso.append(episode_reward_perso)
            # entropy_per_episode_perso.append(episode_entropy_perso)

            # rewards_per_episode_rival.append(episode_reward_rival)
            # entropy_per_episode_rival.append(episode_entropy_rival)

        # print(f"Rewards_per_episode_perso: {rewards_per_episode_perso}\n",f"Rewards_per_episode_rival: {rewards_per_episode_rival}\n" )

        # self.plot_training_results(rewards_per_episode_perso, entropy_per_episode_perso)
        # self.plot_training_results(rewards_per_episode_rival, entropy_per_episode_rival)

        return self.q_table, rival.q_table

    def plot_training_results(self, rewards, entropy):
        episodes = np.arange(len(rewards))

        # Graphique des récompenses
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards, label="Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.legend()

        # Graphique de l'entropie
        plt.subplot(1, 2, 2)
        plt.plot(episodes, entropy, label="Entropy", color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Entropy")
        plt.title("Entropy of Actions per Episode")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def afficher_q_table(self):
        # compteur = 0
        # for state_action, reward in self.q_table.items():
        #     state , action = state_action
        #     df = pd.DataFrame(state)
        #     print("state : ")
        #     print(df)
        #     print("action")
        #     print(action)
        #     print("reward")
        #     print(reward)
        #     if compteur >5:
        #         break
        #     compteur += 1
        print(len(self.q_table))
        return 0




    def save_q_table(self, filename):
        temp_filename = filename + '.tmp'
        with open(temp_filename, 'wb') as f:
            pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_filename, filename) 



    def load_q_table(self, filename):
        # Charge la Q-table depuis un fichier
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)







# Train
def save_q_table(q_table,path):
    temp_filename = path + '.tmp'
    with open(temp_filename, 'wb') as f:
        pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_filename, path) 


def train_agent(agent, game, episodes, rival):
    """
    Entraîne un agent et retourne les Q-tables de l'agent et éventuellement de son rival.
    """
    return agent.train(game, episodes, rival)

# Fonction pour fusionner les Q-tables
def merge_q_tables(q_tables_list):
    # Fusionner les Q-tables en moyenne, ou tout autre méthode de fusion
    # Exemple simple : moyenne des Q-tables
    avg_q_table = {}
    for q_table in q_tables_list:
        for state_action_pair, value in q_table.items():
            if state_action_pair not in avg_q_table:
                avg_q_table[state_action_pair] = value
            if value < avg_q_table[state_action_pair]:
                avg_q_table[state_action_pair] = value
    return avg_q_table

def parallel_train(agents, game, episodes=1000, rival=None):
    """
    Entraîne les agents en parallèle et fusionne les Q-tables.
    """
    # Crée un pool de processus avec autant de processus que le nombre d'agents
    with multiprocessing.Pool(processes=len(agents)) as pool:
        # Prépare une fonction partielle avec les paramètres fixes (jeu, épisodes, rival)
        func = partial(train_agent, game=game, episodes=episodes, rival=rival)
        
        # Utilise `pool.map` pour appliquer `func` à chaque agent
        results = pool.map(func, agents)

    # Sépare les Q-tables retournées (résultat de `train_agent`) en deux listes
    print("ça zip")
    q_tables, rival_q_tables = zip(*results)

    all_q_tables = q_tables + rival_q_tables

    # Fusionne les Q-tables
    merged_q_table = merge_q_tables(all_q_tables)
    
    return merged_q_table

game = QuixoLogic(N=7)
for i in range (20):
    print(i+1)
    agent1 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.5, type_player="X")
    agent2 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.5, type_player="O")
    agent1.load_q_table("../agent_qlearning_save/q_table_agent_sans_multi.pkl")
    agent2.load_q_table("../agent_qlearning_save/q_table_agent_sans_multi.pkl")
    agent1.afficher_q_table()
    q_tables, rival_q_tables = agent1.train(game, episodes=50, rival=agent2)
    all_q_tables = (q_tables, rival_q_tables)
    merged_q_table = merge_q_tables(all_q_tables)
    save_q_table(q_table=merged_q_table, path="../agent_qlearning_save/q_table_agent_sans_multi.pkl")

# 8988

# En utilisant le multiprocessing 
# En utilisant le multiprocessing 
# En utilisant le multiprocessing 
# En utilisant le multiprocessing 

# # Créez vos instances d'agents
# agent1 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.5, type_player="X")
# agent2 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.5, type_player="O")
# # agent3 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.99999, type_player="O")
# # agent4 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.99999, type_player="O")
# # agent5 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.99999, type_player="O")
# # agent6 = QLearningAgent(alpha=1, gamma=0.992, epsilon=0.99999, type_player="O")
# agent1.load_q_table("../agent_qlearning_save/q_table_agent.pkl")
# agent2.load_q_table("../agent_qlearning_save/q_table_agent.pkl")
# # agents = [agent2, agent3, agent4, agent5, agent6]
# agents = [agent2, agent2, agent2]

#     # Définissez le jeu
# game = QuixoLogic(N=7)

#     # Entraînez les agents en parallèle et fusionnez les Q-tables
# merged_q_table = parallel_train(agents, game, episodes=10, rival=agent1)

# # print(merged_q_table)

# save_q_table(q_table=merged_q_table, path="../agent_qlearning_save/q_table_agent.pkl")






# # # # JEU
# root = tk.Tk()
# game = QuixoGame(root)  
# root.mainloop()
