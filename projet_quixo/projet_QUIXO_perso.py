import tkinter as tk
from tkinter import messagebox

# Taille du plateau avec les flèches
N = 7

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

        self.status_label = tk.Label(root, text=f"Joueur courant : {self.current_player}", font=('Arial', 18))
        self.status_label.pack(side=tk.RIGHT, padx=20)

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
        arrow_positions.append((0, j))
        if i == 1:
            arrow_positions.remove((0, j))
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
            self.buttons[x][y].config(text=arrow_direction, command=lambda: self.move_piece(x, y, arrow_direction))
            # Stocker les positions des boutons flèche pour les effacer plus tard
            self.arrow_buttons.append((x, y))

    # Efface les flèches de direction
    def clear_arrows(self):
        for (x, y) in self.arrow_buttons:
            self.buttons[x][y].config(text="")  # Réinitialiser le texte des boutons
        self.arrow_buttons = []

    # Déplacer la pièce selon la direction choisie
    def move_piece(self, x, y, arrow_direction):
        i, j = self.selected_position
        if arrow_direction == "↓":
            self.push_column_down(j)
        elif arrow_direction == "↑":
            self.push_column_up(j)
        elif arrow_direction == "→":
            self.push_row_right(i)
        elif arrow_direction == "←":
            self.push_row_left(i)
        
        # Réinitialiser la sélection après le déplacement
        self.clear_selection()
        if self.check_victory():
            messagebox.showinfo("Victoire", f"Le joueur {self.current_player} a gagné!")
            self.reset_game()
        else:
            self.switch_player()

    def push_column_down(self, col):
        row = self.selected_position[0]
        for i in range(row,1,-1):
            self.board[i][col] = self.board[i-1][col]
            self.buttons[i][col].config(text=self.board[i][col])
        self.board[1][col] = self.current_player
        self.buttons[1][col].config(text=self.current_player)
        

    def push_column_up(self, col):
        row = self.selected_position[0]
        for i in range(row,N-1):
            self.board[i][col] = self.board[i+1][col]
            self.buttons[i][col].config(text=self.board[i][col])
        self.board[N-2][col] = self.current_player
        self.buttons[N-2][col].config(text=self.current_player)


    def push_row_right(self, row):
        col = self.selected_position[1]
        for j in range(col,1,-1):
            self.board[row][j] = self.board[row][j-1]
            self.buttons[row][j].config(text=self.board[row][j])
        self.board[row][1] = self.current_player
        self.buttons[row][1].config(text=self.current_player)


    def push_row_left(self, row):
        col = self.selected_position[1]
        for j in  range(col,N-1):
            self.board[row][j] = self.board[row][j+1]
            self.buttons[row][j].config(text=self.board[row][j])
        self.board[row][N-2] = self.current_player
        self.buttons[row][N-2].config(text=self.current_player)

    def check_victory(self):
    # Vérifie les lignes horizontales
        for i in range(N):
            for j in range(N - 4):
                if all(self.board[i][j + k] == self.current_player for k in range(5)):
                    return True

    # Vérifie les lignes verticales
        for j in range(N):
            for i in range(N - 4):
                if all(self.board[i + k][j] == self.current_player for k in range(5)):
                    return True

    # Vérifie les diagonales descendantes
        for i in range(N - 4):
            for j in range(N - 4):
                if all(self.board[i + k][j + k] == self.current_player for k in range(5)):
                    return True

    # Vérifie les diagonales montantes
        for i in range(4, N):
            for j in range(N - 4):
                if all(self.board[i - k][j + k] == self.current_player for k in range(5)):
                    return True

    # Aucune victoire détectée
        return False


    def switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"
        self.status_label.config(text=f"Joueur courant : {self.current_player}")

    def reset_game(self):
        self.board = [["" for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                self.buttons[i][j].config(text="")
        self.selected_position = None
        self.clear_selection()
        self.current_player = "X"  # Recommence avec le joueur X
        self.status_label.config(text=f"Joueur courant : {self.current_player}")

# Création de la fenêtre principale
root = tk.Tk()
game = QuixoGame(root)
root.mainloop()

