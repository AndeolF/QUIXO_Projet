import tkinter as tk
from tkinter import messagebox

# Taille du plateau avec les flèches
N =7

# Classe pour le jeu Quixo
class QuixoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Jeu Quixo")
        
        # Initialisation du plateau
        self.board = [["" for _ in range(N)] for _ in range(N)]
        self.current_player = "X"
        self.selected_position = None
        
        # Création du canevas principal
        self.canvas = tk.Canvas(root, width=900, height=900)
        self.canvas.pack()

        # Création des boutons pour représenter le plateau de jeu
        self.buttons = [[None for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                btn = tk.Button(root, text="", font=('Arial', 24), width=4, height=2,
                                command=lambda i=i, j=j: self.select_square(i, j))
                btn_window = self.canvas.create_window(100 + j*80, 100 + i*80, window=btn)
                self.buttons[i][j] = btn
        
        # Les boutons pour les flèches
        self.arrow_buttons = []

    # Sélectionner une case
    def select_square(self, i, j):
        if self.is_valid_selection(i, j):
            self.clear_selection()  # Effacer toute sélection précédente
            self.selected_position = (i, j)
            self.buttons[i][j].config(bg='gray')
            self.show_arrows(i, j)

    # Efface la sélection
    def clear_selection(self):
        if self.selected_position:
            i, j = self.selected_position
            self.buttons[i][j].config(bg='light gray')  # Remplacer 'SystemButtonFace' par 'light gray'
            self.selected_position = None
        self.clear_arrows()

    @staticmethod
    def est_premier_carre_interieur(i, j, N):
        # Conditions pour être sur le premier carré intérieur
        if i == 1 and 1 <= j <= N-2:
            return True
        if i == N-2 and 1 <= j <= N-2:
            return True
        if j == 1 and 2 <= i <= N-3:
            return True
        if j == N-2 and 2 <= i <= N-3:
            return True
        return False

    # Vérifie si la case sélectionnée est valide
    def is_valid_selection(self, i, j):
        return (QuixoGame.est_premier_carre_interieur(i, j, N)) and (self.board[i][j] == "" or self.board[i][j] == self.current_player)

    # Affiche les flèches de direction autour du plateau
    def show_arrows(self, i, j):
        N = len(self.board)  # Taille du plateau
        # Créer une liste pour stocker les positions des flèches
        arrow_positions = []
        # Toujours afficher les flèches en (N-1, j) et (i, N-1)
        arrow_positions.append((N-1, j))
        arrow_positions.append((i, N-1))
        arrow_positions.append((i, 0))
        arrow_positions.append((0,j))
        if i == 1:
            arrow_positions.remove((0,j))
        if i == N-2:
            arrow_positions.remove((N-1, j))
        if j == 1:
            arrow_positions.remove((i, 0))
        if j == N-2:
            arrow_positions.remove((i, N-1))
        # Afficher les flèches aux positions déterminées
        for (x, y) in arrow_positions:
            self.update_button_with_arrow(x, y)

    # Mettre à jour le texte du bouton pour afficher la flèche
    def update_button_with_arrow(self, x, y):
        arrow_direction = None
    
        # Déterminer la direction de la flèche en fonction de la position
        if x == 0:  # Flèche vers le bas
            arrow_direction = "↓"
        elif x == len(self.board) - 1:  # Flèche vers le haut
            arrow_direction = "↑"
        elif y == 0:  # Flèche vers la droite
            arrow_direction = "→"
        elif y == len(self.board) - 1:  # Flèche vers la gauche
            arrow_direction = "←"
    
        # Mettre à jour le texte du bouton correspondant pour afficher la flèche
        if arrow_direction:
            self.buttons[x][y].config(text=arrow_direction)
            # Stocker les positions des boutons flèche pour les effacer plus tard
            self.arrow_buttons.append((x, y))

    # Efface les flèches de direction
    def clear_arrows(self):
        for (x, y) in self.arrow_buttons:
            self.buttons[x][y].config(text="")  # Réinitialiser le texte des boutons
        self.arrow_buttons = []


# Création de la fenêtre principale
root = tk.Tk()
game = QuixoGame(root)
root.mainloop()
