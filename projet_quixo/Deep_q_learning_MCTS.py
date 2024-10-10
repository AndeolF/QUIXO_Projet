import copy
import math
import multiprocessing
import os
import pickle
import pickletools
import random
import time
import tkinter as tk

# import tkinter as tk
from collections import defaultdict, deque, namedtuple
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from tkinter import messagebox


class QuixoLogic:
    def __init__(self, N=7):
        self.N = N
        self.board = [["" for _ in range(N)] for _ in range(N)]
        self.current_player = "X"
        self.selected_position = None

    def is_valid_selection(self, i, j):
        return (self.est_premier_carre_interieur(i, j)) and (
            self.board[i][j] == "" or self.board[i][j] == self.current_player
        )

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
        arrows_positions.append((self.N - 1, j))
        arrows_positions.append((i, self.N - 1))
        arrows_positions.append((i, 0))
        arrows_positions.append((0, j))
        if i == 1:
            arrows_positions.remove((0, j))
        if i == self.N - 2:
            arrows_positions.remove((self.N - 1, j))
        if j == 1:
            arrows_positions.remove((i, 0))
        if j == self.N - 2:
            arrows_positions.remove((i, self.N - 1))
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
        for player in ["X", "O"]:  # Vérifie la victoire pour les deux joueurs
            for i in range(self.N):
                for j in range(self.N - 4):
                    if all(self.board[i][j + k] == player for k in range(5)):
                        return True

            for j in range(self.N):
                for i in range(self.N - 4):
                    if all(self.board[i + k][j] == player for k in range(5)):
                        return True

            for i in range(self.N - 4):
                for j in range(self.N - 4):
                    if all(self.board[i + k][j + k] == player for k in range(5)):
                        return True

            for i in range(4, self.N):
                for j in range(self.N - 4):
                    if all(self.board[i - k][j + k] == player for k in range(5)):
                        return True

        return False

    def get_winner(self):
        winner_X = False
        winner_O = False

        # Vérifie si "X" ou "O" ont gagné
        for player in ["X", "O"]:
            for i in range(self.N):
                for j in range(self.N - 4):
                    if all(self.board[i][j + k] == player for k in range(5)):
                        if player == "X":
                            winner_X = True
                        else:
                            winner_O = True

            for j in range(self.N):
                for i in range(self.N - 4):
                    if all(self.board[i + k][j] == player for k in range(5)):
                        if player == "X":
                            winner_X = True
                        else:
                            winner_O = True

            for i in range(self.N - 4):
                for j in range(self.N - 4):
                    if all(self.board[i + k][j + k] == player for k in range(5)):
                        if player == "X":
                            winner_X = True
                        else:
                            winner_O = True

            for i in range(4, self.N):
                for j in range(self.N - 4):
                    if all(self.board[i - k][j + k] == player for k in range(5)):
                        if player == "X":
                            winner_X = True
                        else:
                            winner_O = True

        # Si les deux joueurs gagnent en même temps, retourner le joueur qui n'est pas current_player
        if winner_X and winner_O:
            return "O" if self.current_player == "X" else "X"
        elif winner_X:
            return "X"
        elif winner_O:
            return "O"
        else:
            return None  # Pas de gagnant

    def is_board_full(self):
        for row in extract_state_from_board(self.board):
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

    def get_available_actions(self):
        # Retourne une liste des actions possibles pour l'IA
        actions = []
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if self.is_valid_selection(i, j):
                    arrows_postions = self.give_arrows_position(i, j)
                    for arrow_postions in arrows_postions:
                        direction = self.give_one_arrow_direction(arrow_postions)
                        actions.append((i, j, direction))
        return actions

    def step(self, action, type_player):
        i, j, arrow_direction = action
        done = False
        self.move_piece(arrow_direction, i, j)
        if self.check_victory():
            done = True
        if self.is_board_full():
            done = True
        self.switch_player()
        next_state = self.board  # L'état après l'action
        reward = self.compute_reward_3(type_player)  # Calculer la récompense
        return next_state, reward, done

    def compute_reward_3(self, type_player):
        state = extract_state_from_board(self.board)

        opponent = "O" if type_player == "X" else "X"

        def longest_sequence(player, board):
            pattern_score = 0
            sequence_score = 0
            max_len = 0
            position_score = 0

            # Vérifier les lignes
            for row in board:
                sequence_score += sequence_in_line(row, player)
                max_len = max(max_len, max_in_line(row, player))
                pattern_score += pattern_in_line(row, player)
                position_score += position_in_line(row, player)

            # Vérifier les colonnes
            for col in range(len(board)):
                column = [board[row][col] for row in range(len(board))]
                sequence_score += sequence_in_line(column, player)
                max_len = max(max_len, max_in_line(column, player))
                pattern_score += pattern_in_line(column, player)
                position_score += position_in_line(column, player)

            # Vérifier les diagonales
            diagonals = get_diagonals(board)
            for diagonal in diagonals:
                sequence_score += sequence_in_line(diagonal, player)
                max_len = max(max_len, max_in_line(diagonal, player))
                pattern_score += pattern_in_line(diagonal, player)
                position_score += position_in_line(diagonal, player)

            return max_len, sequence_score, pattern_score, position_score

        def position_in_line(line, player):
            max_len = 0
            current_len = 0
            for cell in line:
                if cell == player:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0
            if max_len > 2:
                return 0
            else:
                compteur_symbole = 0
                for cell in line:
                    if cell == player:
                        compteur_symbole += 1
                if compteur_symbole > 2:
                    return 5
                else:
                    return 0

        def sequence_in_line(line, player):
            max_len = 0
            current_len = 0
            for cell in line:
                if cell == player:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0
            if max_len < 2:
                return 0
            if max_len == 2:
                return 1.5
            if max_len == 3:
                return 5
            if max_len == 4:
                return 20
            else:
                return 0

        def max_in_line(line, player):
            max_len = 0
            current_len = 0
            for cell in line:
                if cell == player:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0
            else:
                return max_len

        def pattern_in_line(line, player):
            oppo = "O" if player == "X" else "X"
            point_pattern = 0
            point_si_vide = 20
            point_si_pas_vide = 15

            # Cas : Deux symboles de chaque côté et un trou au milieu
            if (
                line[0] == player
                and line[1] == player
                and line[3] == player
                and line[4] == player
            ):
                if line[2] == "":  # Trou vide au milieu
                    point_pattern += point_si_vide
                elif line[2] == oppo:  # Symbole adverse au milieu
                    point_pattern += point_si_pas_vide

            # Cas : Trois symboles d'un côté et un trou près de l'autre extrémité (à droite)
            if (
                line[0] == player
                and line[1] == player
                and line[2] == player
                and line[4] == player
            ):
                if line[3] == "":  # Trou vide avant l'extrémité
                    point_pattern += point_si_vide
                elif line[3] == oppo:  # Symbole adverse avant l'extrémité
                    point_pattern += point_si_pas_vide

            # Cas : Trois symboles d'un côté et un trou près de l'autre extrémité (à gauche)
            if (
                line[4] == player
                and line[3] == player
                and line[2] == player
                and line[0] == player
            ):
                if line[1] == "":  # Trou vide avant l'extrémité opposée
                    point_pattern += point_si_vide
                elif line[1] == oppo:  # Symbole adverse avant l'extrémité opposée
                    point_pattern += point_si_pas_vide

            # Cas : Quatre symboles consécutifs avec trou à l'extrémité (droite)
            if (
                line[0] == player
                and line[1] == player
                and line[2] == player
                and line[3] == player
            ):
                if line[4] == "":  # Trou vide à la fin
                    point_pattern += 0.5

            # Cas : Quatre symboles consécutifs avec trou à l'extrémité (gauche)
            if (
                line[1] == player
                and line[2] == player
                and line[3] == player
                and line[4] == player
            ):
                if line[0] == "":  # Trou vide au début
                    point_pattern += 0.5

            return point_pattern

        def get_diagonals(board):
            diagonals = []
            n = len(board)
            # Diagonale principale
            diagonals.append([board[i][i] for i in range(n)])
            # Diagonale secondaire
            diagonals.append([board[i][n - 1 - i] for i in range(n)])

            return diagonals

        def central_control(board, player):
            central_positions = [
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
            ]
            central_weight = 0.5  # Poids pour les positions centrales
            control_score = 0

            for x, y in central_positions:
                if board[x][y] == player:
                    control_score += central_weight

            return control_score

        # Calculer les suites les plus longues, le potentiel d'alignement et les motifs pour les deux joueurs
        player_max_len, player_sequence_score, player_pattern, player_position = (
            longest_sequence(type_player, state)
        )
        (
            opponent_max_len,
            opponent_sequence_score,
            opponent_pattern,
            opponent_position,
        ) = longest_sequence(opponent, state)

        # Calculer le contrôle central pour les deux joueurs
        player_central_control = central_control(state, type_player)
        opponent_central_control = central_control(state, opponent)

        # max len sert à savoir si le joueur à gagné ou perdu
        if opponent_max_len == 5:
            return -100
        if player_max_len == 5:
            return 100

        # Calculer le score final en combinant la séquence maximale, le potentiel, les motifs et le contrôle central
        player_score = (
            player_sequence_score
            + player_pattern
            + player_position
            + player_central_control
        )
        opponent_score = (
            opponent_sequence_score
            + opponent_pattern
            + opponent_position
            + opponent_central_control
        ) * 2
        # On avantage l'opposant pour forcer une réaction plus défensive de l'agent et parceque à ce moment la, après le calcule de la reward c'est à l'opposant de jouer donc il a un avantage pris en compte ici

        return player_score - opponent_score


class QuixoGame:
    def __init__(self, root, logic=QuixoLogic(), agent=None, agent_type_player="O"):
        self.root = root
        self.root.title("Jeu Quixo")
        self.agent = agent
        self.logic = logic
        self.selected_position = None
        self.agent_type_player = agent_type_player

        self.canvas = tk.Canvas(root, width=900, height=900)
        self.canvas.pack()

        self.status_label = tk.Label(
            root,
            text=f"Joueur courant : {self.logic.current_player}",
            font=("Arial", 18),
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)

        self.buttons = [
            [None for _ in range(self.logic.N)] for _ in range(self.logic.N)
        ]
        for i in range(self.logic.N):
            for j in range(self.logic.N):
                btn = tk.Button(
                    root,
                    text="",
                    font=("Arial", 24),
                    width=4,
                    height=2,
                    command=lambda i=i, j=j: self.select_square(i, j),
                )
                btn_window = self.canvas.create_window(
                    100 + j * 80, 100 + i * 80, window=btn
                )
                self.buttons[i][j] = btn

        self.arrow_buttons = []

        if self.agent:
            self.agent.model.eval()

        if self.agent and self.logic.current_player == self.agent_type_player:
            self.root.after(500, self.ai_play)

    def select_square(self, i, j):
        if self.logic.is_valid_selection(i, j):
            self.clear_selection()
            self.selected_position = (i, j)
            self.buttons[i][j].config(bg="gray")
            self.show_arrows(i, j)

    def show_arrows(self, i, j):
        arrows_positions = self.logic.give_arrows_position(i, j)
        for arrow_position in arrows_positions:
            x, y = arrow_position
            arrow_direction = self.logic.give_one_arrow_direction(arrow_position)
            self.buttons[x][y].config(
                text=arrow_direction, command=lambda x=x, y=y: self.select_arrow(x, y)
            )
            self.arrow_buttons.append((x, y))

    def select_arrow(self, x, y):
        arrow_direction = self.buttons[x][y].cget("text")
        if arrow_direction in ["↓", "↑", "→", "←"]:
            i, j = self.selected_position
            self.logic.move_piece(arrow_direction, i, j)
            self.clear_selection()
            self.update_board()
            flag = self.stoper_si_fin()
            self.update_status()
            if not flag:
                self.logic.switch_player()
                if self.agent and self.logic.current_player == self.agent_type_player:
                    self.root.after(10, self.ai_play)

    def stoper_si_fin(self):
        if self.logic.check_victory():
            winner = self.logic.get_winner()
            self.root.after(250, lambda: self.handle_victory(winner))
            return True
        elif self.logic.is_board_full():
            messagebox.showinfo("Match nul !")
            self.reset_game()
            return True
        return False

    def handle_victory(self, winner):
        messagebox.showinfo("Victoire", f"Le joueur {winner} a gagné!")
        self.reset_game()

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
            self.buttons[i][j].config(bg="light gray")
            self.selected_position = None
        self.clear_arrows()

    def clear_arrows(self):
        for x, y in self.arrow_buttons:
            self.buttons[x][y].config(text="")
        self.arrow_buttons = []

    def reset_game(self):
        self.logic.reset_game()
        self.update_board()
        self.update_status()
        if self.agent and self.logic.current_player == self.agent_type_player:
            self.root.after(500, self.ai_play)

    def ai_play(self):
        start_time = time.time()
        state = extract_state_from_board(self.logic.board)
        action = self.agent.select_action(state, self.logic, use_mcts=True)
        print(f"temps mis pour selec l'action = {time.time() - start_time}")
        next_state, reward, done = self.logic.step(action, self.agent_type_player)
        # print(f"la reward vrai : {reward}\n")
        self.update_board()
        self.stoper_si_fin()
        if done:
            # messagebox.showinfo("Victoire", "L'IA a gagné!")
            return True


# Définition de la classe DQN (réseau principal et cible)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu((self.fc1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu((self.fc2(x)))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return self.fc4(x)


class DQN_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)  # dim constantes pour la connexion résiduelle
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(0.333)

    def forward(self, x):
        # Connexion résiduelle après les deux premières couches

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)

        # Connexion résiduelle après deux couches avec dimensions identiques
        identity_1 = x
        x = torch.nn.functional.relu(self.fc3(x))
        # Remplacer `x += identity` par une addition hors inplace pour créer un nouveau vecteur
        x = x + identity_1  # Pas de modification in-place

        # Continuation du forward pass sans autres connexions résiduelles
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.dropout(x)
        identity_2 = x
        x = torch.nn.functional.relu(self.fc5(x))
        x = x + identity_2

        return self.fc6(x)


# Expérience Replay
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "dones")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_average_rewards_per_episode(rewards):
    average_rewards = [reward for reward in rewards]

    # Afficher les résultats
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(average_rewards)),
        average_rewards,
        label="Moyenne des récompenses par épisode",
    )
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense moyenne")
    plt.title("Moyenne des récompenses par épisode")
    plt.legend()
    plt.show()


def plot_rewards(rewards1, window_size=10):
    """
    Trace l'évolution de deux séries de récompenses dans le temps avec lissage.

    :param rewards1: Liste ou tableau des premières récompenses.
    :param rewards2: Liste ou tableau des secondes récompenses.
    :param window_size: Taille de la fenêtre pour le lissage (par défaut 10).
    """

    def smooth_data(data, window_size):
        """Lisse les données avec une moyenne mobile."""
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Appliquer le lissage aux deux séries de récompenses
    smoothed_rewards1 = smooth_data(rewards1, window_size)

    # Créer une nouvelle figure
    plt.figure(figsize=(10, 6))

    # Tracer les deux séries de récompenses
    plt.plot(
        smoothed_rewards1, label="Rewards", color="blue", linestyle="-", linewidth=2
    )

    # Ajouter un titre et des labels
    plt.title("Évolution des Rewards avec Lissage")
    plt.xlabel("Temps")
    plt.ylabel("Rewards")

    # Ajouter une légende
    plt.legend()

    # Afficher la grille pour une meilleure lisibilité
    plt.grid(True)

    # Afficher le graphique
    plt.show()


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
    if index == 0:
        return (1, 1, "←")  # Coin en haut à gauche
    elif index == 1:
        return (1, 1, "↑")
    elif index == 2:
        return (5, 5, "→")
    elif index == 3:
        return (5, 5, "↓")
    elif index == 4:
        return (5, 1, "←")
    elif index == 5:
        return (1, 5, "↑")
    elif index == 6:
        return (1, 5, "→")
    elif index == 7:
        return (5, 1, "↓")

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
    if len(board[0]) > 5:
        state = [row[1:6] for row in board[1:6]]
        return state
    else:
        return board


def get_available_actions_indices(game):
    available_actions = (
        game.get_available_actions()
    )  # Renvoie une liste de tuples (i, j, direction)
    return [action_to_index(i, j, direction) for (i, j, direction) in available_actions]


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


def decode_state(encoded_state, grid_size):
    decoded_state = []
    for i in range(0, len(encoded_state), 3):
        if encoded_state[i : i + 3] == [1, 0, 0]:
            cell = ""
        elif encoded_state[i : i + 3] == [0, 1, 0]:
            cell = "X"
        elif encoded_state[i : i + 3] == [0, 0, 1]:
            cell = "O"
        # Ajoute la cellule au plateau sous forme de grille
        if len(decoded_state) % grid_size == 0:
            decoded_state.append([cell])
        else:
            decoded_state[-1].append(cell)

    return decoded_state


class BruteForceAgent:
    def __init__(self, model, num_line=0):
        self.model = model
        self.num_line = num_line
        self.epsilon = 0.5

    def select_action(self, epsilon, state, game):
        available_actions = game.get_available_actions()
        if self.num_line < 5:
            for i in range(4, 0, -1):
                action = (i + 1, self.num_line + 1, "↓")
                if action in available_actions:
                    i_, j_, arrow = action
                    if game.board[i_][j_] == "":
                        return action
            return random.choice(available_actions)
        elif self.num_line < 10:
            for i in range(0, 3):
                action = (i + 1, self.num_line - 5 + 1, "↑")
                if action in available_actions:
                    i_, j_, arrow = action
                    if game.board[i_][j_] == "":
                        return action
            return random.choice(available_actions)
        elif self.num_line < 15:
            for i in range(4, 0, -1):
                action = (self.num_line - 10 + 1, i + 1, "→")
                if action in available_actions:
                    i_, j_, arrow = action
                    if game.board[i_][j_] == "":
                        return action
            return random.choice(available_actions)
        elif self.num_line < 10:
            for i in range(0, 3):
                action = (i + 1, self.num_line - 15 + 1, "←")
                if action in available_actions:
                    i_, j_, arrow = action
                    if game.board[i_][j_] == "":
                        return action
            return random.choice(available_actions)
        else:
            return random.choice(available_actions)


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action


class AgentMCTS_DQL:
    def __init__(
        self,
        model,
        optimizer,
        memory,
        num_simulations=100,
        max_depth=10,
        batch_size=64,
        epsilon=0.3,
        gamma=0.9,
        C=1.414,
    ):
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.C = C
        self.target_net = copy.deepcopy(model)
        self.device = "cpu"
        self.model.to(self.device)

    def select_action(self, state, game, use_mcts=True):
        agent_player = game.current_player
        available_actions = game.get_available_actions()
        for action in available_actions:
            game_copy_rapide = copy.deepcopy(game)
            next_board, reward, done = game_copy_rapide.step(
                action, type_player=agent_player
            )
            if reward > 99:
                # print("vitesse")
                return action

        # # Version 2

        if use_mcts:
            # Vérifier si le jeu est terminé
            if self.is_terminal(game):
                return None  # Vous pourriez vouloir renvoyer un indicateur spécial pour les états terminaux

            root_node = MCTSNode(state)
            agent_player = game.current_player
            for simu in range(self.num_simulations):
                node = self.mcts_select(root_node, game, agent_player)
                # il faut maintenant réaligné le node avec le jeu game
                game_copy = copy.deepcopy(game)
                current_node = node
                actions_taken = []

                # # # Remonter dans l'arbre jusqu'à la racine pour retrouver les actions menant à ce nœud et ainsi aligné le game avec le node selectionné
                while current_node.parent is not None:
                    actions_taken.append(current_node.action)
                    current_node = current_node.parent
                if actions_taken != []:
                    actions_taken.reverse()
                    for action in actions_taken:
                        game_copy.step(action, type_player=game_copy.current_player)

                if not self.is_terminal(game_copy):
                    self.mcts_expand(node, game_copy)

                # # # Vérifier si le nœud racine a des enfants après l'expansion
                if not root_node.children:
                    # print("pas enfant")
                    return self.epsilon_greedy_action(state, game_copy)

                q_value = self.mcts_simulate(game_copy, node, agent_player)
                self.mcts_backpropagate(node, q_value)

            # for child in root_node.children:
            #     print(f"value : {child.value}")

            # Choisir l'action correspondant à l'enfant avec la meilleur value (comme le nombre de visite (prise en compte de UCT) et prise en compte dans la backprop)

            best_child = max(root_node.children, key=lambda child: child.value)
            # print(f"best_child value : {best_child.value}")
            return best_child.action
        else:
            # print("pas mcts")
            return self.epsilon_greedy_action(state, game)

    def epsilon_greedy_action(self, state, game):
        available_actions_indices = get_available_actions_indices(game)
        if random.random() < self.epsilon:
            action_index = random.choice(available_actions_indices)
        else:
            with torch.no_grad():
                state_tensor = (
                    torch.tensor(encode_state(state), dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                q_values = self.model(state_tensor)

                valid_q_values = q_values[
                    0, [valid_index_to_faux_index[i] for i in available_actions_indices]
                ]
                action_index = available_actions_indices[valid_q_values.argmax().item()]
        action = index_to_action(action_index)
        return action

    def mcts_select(self, node, game, agent_player):
        # Sélectionne un nœud en fonction de l'UCT

        while node.children:
            node = max(node.children, key=lambda child: self.uct(child, node))
        return node

    def mcts_expand(self, node, game):
        # Étend le nœud avec tous les enfants possibles
        available_actions = game.get_available_actions()
        for action in available_actions:
            # Simuler l'action dans le jeu pour obtenir le nouvel état
            game_copy = copy.deepcopy(game)
            next_state, _, _ = game_copy.step(
                action, type_player=game_copy.current_player
            )
            child_node = MCTSNode(next_state, parent=node, action=action)
            node.children.append(child_node)

    def mcts_simulate(self, game, node, agent_player):
        # Créer une copie du jeu
        game_copy = copy.deepcopy(game)
        available_actions = game.get_available_actions()
        for action in available_actions:
            game_copy_rapide = copy.deepcopy(game_copy)
            next_board, reward, done = game_copy_rapide.step(
                action, type_player=agent_player
            )
            if reward < -99:
                # print("c'est foutu")
                return -400
        # # # print("df_1")
        # # # state_1 = extract_state_from_board(game_copy.board)
        # # # df_1 = pd.DataFrame(state_1)
        # # # print(df_1)

        # # Commencez la simulation à partir de cet état ajusté
        cumulative_reward = 0
        depth = 0

        state = extract_state_from_board(game_copy.board)
        current_state = state
        # print("df_2 :")
        # df_2 = pd.DataFrame(extract_state_from_board(game_copy.board))
        # print(df_2)

        # Boucle jusqu'à atteindre un état terminal ou une profondeur maximale
        nbr_de_coup = 0
        while not self.is_terminal(game_copy) and depth < self.max_depth:
            available_actions = game_copy.get_available_actions()
            if not available_actions:
                print("ça break")
                break  # Si aucune action disponible, on sort de la boucle

            # Sélectionner une action aléatoire ou basée sur une stratégie simple
            action = self.epsilon_greedy_action(current_state, game_copy)
            next_board, reward, done = game_copy.step(action, type_player=agent_player)
            nbr_de_coup += 1

            # Ajouter la récompense intermédiaire à la somme cumulative
            cumulative_reward += reward * ((self.gamma) ** nbr_de_coup)

            if done:
                return cumulative_reward

            next_state = extract_state_from_board(next_board)
            current_state = next_state
            depth += 1

        # Si nous avons atteint la profondeur maximale, estimer la Q-value avec le réseau de neurones
        state_tensor = (
            torch.tensor(self.encode_state(current_state), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.model(state_tensor)
            max_q_value = q_values.max().item()

        # Inverser la Q-value si c'est le tour de l'adversaire
        if game_copy.current_player != agent_player:
            max_q_value = -max_q_value

        # Retourner la somme des récompenses cumulées + la Q-value estimée
        return cumulative_reward + max_q_value

    def mcts_backpropagate(self, node, q_value):
        # Rétro-propage la récompense à travers l'arbre
        while node is not None:
            node.visits += 1
            node.value += (q_value - node.value) / node.visits
            node = node.parent

    def uct(self, child, parent):
        # Upper Confidence Bound for Trees (UCT)
        return child.value / (child.visits + 1e-6) + self.C * np.sqrt(
            np.log(parent.visits + 1) / (child.visits + 1e-6)
        )

    def optimize_model(self):
        # Optimisation classique DQL avec la mémoire de replay
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.dones).to(self.device)

        # Q(s, a) pour les actions prises
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Calcul des valeurs Q pour les prochains états
        next_state_actions = self.model(next_state_batch).argmax(dim=1, keepdim=True)
        next_state_values = (
            self.target_net(next_state_batch).gather(1, next_state_actions).detach()
        )

        next_state_values = next_state_values * (1 - done_batch.unsqueeze(1))

        # Valeurs Q attendues
        expected_state_action_values = reward_batch.unsqueeze(1) + (
            self.gamma * next_state_values
        )

        # Calcul de la perte
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.model.state_dict())

    # encodage one-hot
    @staticmethod
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

    def is_terminal(self, game):
        return game.check_victory() or game.is_board_full()


def train_agent_with_mcts_dql(
    game,
    agent,
    num_episodes,
    decroissance_epsilon=True,
    TEST=False,
):
    agent.model.train()  # Met le modèle en mode entraînement
    tab_reward_total = []

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state = extract_state_from_board(game.reset_game())
        state_tensor = (
            torch.tensor(encode_state(state), dtype=torch.float32)
            .unsqueeze(0)
            .to(agent.device)
        )

        total_reward = 0

        for t in range(256):  # Limite le nombre d'étapes par épisode
            # Utiliser MCTS pour choisir l'action
            # print(f"\nJoueur : {game.current_player}")
            # df_1 = pd.DataFrame(state)
            # print(df_1)
            action = agent.select_action(state, game, use_mcts=True)
            if action is None:
                break
            # print(f"action : {action}")
            # Appliquer l'action et récupérer les résultats du jeu
            if t % 2 == 0:
                t_p = "X"
            else:
                t_p = "O"
            next_board, reward, done = game.step(action, type_player=t_p)
            # print(f"reward : {reward}")
            next_state = extract_state_from_board(next_board)
            next_state_tensor = (
                torch.tensor(encode_state(next_state), dtype=torch.float32)
                .unsqueeze(0)
                .to(agent.device)
            )

            # Mise à jour des récompenses et état
            total_reward += reward
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(agent.device)
            done_tensor = torch.tensor([1.0 if done else 0.0], dtype=torch.float32).to(
                agent.device
            )

            if not TEST:
                # Stocker la transition dans la mémoire de l'agen
                i, j, direction = action
                agent.memory.push(
                    state_tensor,
                    torch.tensor(
                        [[valid_index_to_faux_index[action_to_index(i, j, direction)]]],
                        dtype=torch.long,
                    ).to(agent.device),
                    next_state_tensor,
                    reward_tensor,
                    done_tensor,
                )

            if not TEST:
                # Optimiser le modèle via DQL
                agent.optimize_model()

            # Passer à l'état suivant
            state = next_state
            state_tensor = next_state_tensor
            # df_1 = pd.DataFrame(state)
            # print(df_1)

            # Si la partie est terminée
            if done:
                break

        # Sauvegarde des récompenses cumulées pour cet épisode
        tab_reward_total.append(total_reward)

        # Décroissance d'exploration (epsilon)
        if decroissance_epsilon:
            agent.epsilon = max(
                0.33, agent.epsilon * 0.99
            )  # Décroît epsilon progressivement

        # Mise à jour périodique du réseau cible
        if episode % 10 == 0:
            agent.update_target_network()

    print(agent.epsilon)

    return tab_reward_total


input_dim = 5 * 5 * 3  # 5x5 plateau et 3 canaux pour "", "X", "O"
learning_rate = 0.0001

# Initialisation des paramètres et du réseau
valid_indices = [
    1,
    0,
    9,
    8,
    10,
    13,
    12,
    14,
    17,
    16,
    18,
    5,
    6,
    33,
    32,
    35,
    45,
    46,
    47,
    37,
    36,
    39,
    49,
    50,
    51,
    41,
    40,
    43,
    53,
    54,
    55,
    4,
    7,
    20,
    22,
    23,
    24,
    26,
    27,
    28,
    30,
    31,
    2,
    3,
]

output_dim = len(valid_indices)  # 44
faux_index_to_valid_index = {
    i: valid_index for i, valid_index in enumerate(valid_indices)
}
valid_index_to_faux_index = {
    valid_index: i for i, valid_index in enumerate(valid_indices)
}


policy_net_1 = DQN_2(input_dim, output_dim)
policy_net_2 = DQN_2(input_dim, output_dim)

policy_net_1.load_state_dict(
    torch.load("./model_on_train_agent_DQL_MCTS.pth", weights_only=True)
)

# policy_net_2.load_state_dict(
#     torch.load("../models/model_on_train_agent_2.pth", weights_only=True)
# )


optimizer_1 = optim.AdamW(
    policy_net_1.parameters(), lr=learning_rate, weight_decay=1e-3
)
optimizer_2 = optim.AdamW(
    policy_net_2.parameters(), lr=learning_rate, weight_decay=1e-3
)

memory_1 = ReplayMemory(10000)
memory_2 = ReplayMemory(10000)

# Agent 1 est pour les "X"
# agent_1 = AgentDQL(
#     policy_net_1, optimizer_1, batch_size=64, epsilon=0.05, gamma=0.9, memory=memory_1
# )
# # Agent 2 est pour les "O"
# agent_2 = AgentDQL(
#     policy_net_2, optimizer_2, batch_size=64, epsilon=0.7, gamma=0.9, memory=memory_2
# )

agent_3 = BruteForceAgent(policy_net_1)


game = QuixoLogic(N=7)


# agent_DQL_MCTS = AgentMCTS_DQL(
#     policy_net_1,
#     optimizer_1,
#     memory_1,
#     num_simulations=500,
#     max_depth=100,
#     batch_size=64,
#     epsilon=0.2,
#     gamma=0.8,
#     C=1.414,
# )

# tab_reward_total = train_agent_with_mcts_dql(
#     game,
#     agent_DQL_MCTS,
#     num_episodes=500,
#     decroissance_epsilon=True,
#     TEST=False,
# )

# print(tab_reward_total)
# plot_rewards(tab_reward_total, window_size=2)


# To save
# To save
# To save
# To save

# # # Avec un modèle DQN_2 avec 2 connexion residuel pour un agent MCTS_DQL :
# torch.save(
#     agent_DQL_MCTS.model.state_dict(), "../models/model_on_train_agent_DQL_MCTS.pth"
# )

# Test avec DQN_2 avec 2 connexion residuel (identity_1 et identity_2) :
# torch.save(agent_2.model.state_dict(), "../models/model_on_train_agent_1.pth")
# torch.save(agent_2.model.state_dict(), "../models/model_on_train_agent_2.pth")

# Test avec DQN_2 avec une seul connexion residuel (identity_1) :
# torch.save(agent_1.model.state_dict(), "../models/model_vraiment_bof_agent1.pth")
# torch.save(agent_2.model.state_dict(), "../models/model_vraiment_bof_agent2.pth")

# Tout ceux ci dessous sont des DQN

# les mieux pour le moment :
# torch.save(agent_1.model.state_dict(), "../models/model_vraiment_bien_agent1.pth")
# torch.save(agent_2.model.state_dict(), "../models/model_vraiment_bien_agent2.pth")


# torch.save(agent_1.model.state_dict(), "../models/model_bien_agent1.pth")
# torch.save(agent_2.model.state_dict(), "../models/model_bien_agent2.pth")


# torch.save(agent_2.model.state_dict(), "../models/model_78_agent2.pth")
# torch.save(agent_1.model.state_dict(), "../models/model_94.pth")
# torch.save(agent_1.model.state_dict(), "../models/model_agent1_99.pth")
# torch.save(agent_2.model.state_dict(), "../models/model_agent2_99.pth")


# Jouer avec le l'interface :
# Jouer avec le l'interface :
# Jouer avec le l'interface :
# Jouer avec le l'interface :
# Jouer avec le l'interface :
# Jouer avec le l'interface :

# Paramètre pour un agent fort (si on réduit num_simulation, cela réduit la force de l'agent(de manière non linéaire cependant)) :

agent_DQL_MCTS = AgentMCTS_DQL(
    policy_net_1,
    optimizer_1,
    memory_1,
    num_simulations=500,
    max_depth=100,
    batch_size=64,
    epsilon=0.2,
    gamma=0.8,
    C=1.414,
)

root = tk.Tk()
game_logic = QuixoLogic(N=7)
# game_interface = QuixoGame(
#     root=root, logic=game_logic, agent=agent_DQL_MCTS, agent_type_player="O"
# )
game_interface = QuixoGame(
    root=root, logic=game_logic, agent=agent_DQL_MCTS, agent_type_player="O"
)
root.mainloop()
