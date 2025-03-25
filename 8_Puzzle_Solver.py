import time
import heapq
import tkinter as tk
from tkinter import messagebox, ttk
import random
from collections import deque

# Class to represent a state of the puzzle
class PuzzleState:
    def __init__(self, board, parent=None, move=None, depth=0):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this state
        self.depth = depth
        
        # For A* and Greedy
        self.f = 0  # f(n) = g(n) + h(n)
        self.g = depth  # g(n) = depth (cost so far)
        self.h = 0  # h(n) = heuristic (estimated cost to goal)
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(str(self.board))
    
    def get_blank_position(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j
    
    def get_possible_moves(self):
        moves = []
        i, j = self.get_blank_position()
        
        # Check all four directions
        if i > 0:
            moves.append('Up')
        if i < 2:
            moves.append('Down')
        if j > 0:
            moves.append('Left')
        if j < 2:
            moves.append('Right')
            
        return moves
    
    def get_new_state(self, move):
        i, j = self.get_blank_position()
        new_board = [row[:] for row in self.board]  # Deep copy
        
        if move == 'Up':
            new_board[i][j], new_board[i-1][j] = new_board[i-1][j], new_board[i][j]
        elif move == 'Down':
            new_board[i][j], new_board[i+1][j] = new_board[i+1][j], new_board[i][j]
        elif move == 'Left':
            new_board[i][j], new_board[i][j-1] = new_board[i][j-1], new_board[i][j]
        elif move == 'Right':
            new_board[i][j], new_board[i][j+1] = new_board[i][j+1], new_board[i][j]
            
        return PuzzleState(new_board, self, move, self.depth + 1)
    
    def calculate_manhattan_distance(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    # Calculate the position where this tile should be
                    value = self.board[i][j]
                    goal_i, goal_j = (value-1) // 3, (value-1) % 3
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def calculate_misplaced_tiles(self):
        count = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0 and self.board[i][j] != i*3 + j + 1:
                    count += 1
        return count
    
    def is_goal(self):
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.board == goal

# Solver class to implement different algorithms
class PuzzleSolver:
    def __init__(self, initial_board):
        self.initial_state = PuzzleState(initial_board)
        self.goal_state = PuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    def get_solution_path(self, final_state):
        path = []
        current = final_state
        while current.parent:
            path.append(current.move)
            current = current.parent
        return path[::-1]  # Reverse to get path from start to goal
    
    def bfs(self):
        start_time = time.time()
        queue = deque([self.initial_state])
        visited = set([str(self.initial_state.board)])
        
        while queue:
            state = queue.popleft()
            
            if state.is_goal():
                end_time = time.time()
                return {
                    'solution': self.get_solution_path(state),
                    'time_taken': end_time - start_time
                }
            
            for move in state.get_possible_moves():
                new_state = state.get_new_state(move)
                state_str = str(new_state.board)
                
                if state_str not in visited:
                    visited.add(state_str)
                    queue.append(new_state)
        
        end_time = time.time()
        return {'solution': None, 'time_taken': end_time - start_time}
    
    def dfs(self, max_depth=20):
        start_time = time.time()
        stack = [(self.initial_state, set([str(self.initial_state.board)]))]
        
        while stack:
            state, visited = stack.pop()
            
            if state.is_goal():
                end_time = time.time()
                return {
                    'solution': self.get_solution_path(state),
                    'time_taken': end_time - start_time
                }
            
            if state.depth < max_depth:
                for move in reversed(state.get_possible_moves()):  # Reverse for consistent exploration
                    new_state = state.get_new_state(move)
                    state_str = str(new_state.board)
                    
                    if state_str not in visited:
                        new_visited = visited.copy()
                        new_visited.add(state_str)
                        stack.append((new_state, new_visited))
        
        end_time = time.time()
        return {'solution': None, 'time_taken': end_time - start_time}
    
    def ids(self, max_depth=30):
        start_time = time.time()
        
        for depth in range(max_depth):
            stack = [(self.initial_state, set([str(self.initial_state.board)]))]
            
            while stack:
                state, visited = stack.pop()
                
                if state.is_goal():
                    end_time = time.time()
                    return {
                        'solution': self.get_solution_path(state),
                        'time_taken': end_time - start_time
                    }
                
                if state.depth < depth:
                    for move in reversed(state.get_possible_moves()):
                        new_state = state.get_new_state(move)
                        state_str = str(new_state.board)
                        
                        if state_str not in visited:
                            new_visited = visited.copy()
                            new_visited.add(state_str)
                            stack.append((new_state, new_visited))
        
        end_time = time.time()
        return {'solution': None, 'time_taken': end_time - start_time}
    
    def a_star(self):
        start_time = time.time()
        
        self.initial_state.h = self.initial_state.calculate_manhattan_distance()
        self.initial_state.f = self.initial_state.g + self.initial_state.h
        
        open_set = [self.initial_state]
        closed_set = set()
        
        while open_set:
            state = heapq.heappop(open_set)
            
            if state.is_goal():
                end_time = time.time()
                return {
                    'solution': self.get_solution_path(state),
                    'time_taken': end_time - start_time
                }
            
            state_str = str(state.board)
            if state_str in closed_set:
                continue
                
            closed_set.add(state_str)
            
            for move in state.get_possible_moves():
                new_state = state.get_new_state(move)
                new_state_str = str(new_state.board)
                
                if new_state_str in closed_set:
                    continue
                
                new_state.h = new_state.calculate_manhattan_distance()
                new_state.f = new_state.g + new_state.h
                
                heapq.heappush(open_set, new_state)
        
        end_time = time.time()
        return {'solution': None, 'time_taken': end_time - start_time}
    
    def greedy(self):
        start_time = time.time()
        
        self.initial_state.h = self.initial_state.calculate_misplaced_tiles()
        self.initial_state.f = self.initial_state.h  # f = h for greedy
        
        open_set = [self.initial_state]
        closed_set = set()
        
        while open_set:
            state = heapq.heappop(open_set)
            
            if state.is_goal():
                end_time = time.time()
                return {
                    'solution': self.get_solution_path(state),
                    'time_taken': end_time - start_time
                }
            
            state_str = str(state.board)
            if state_str in closed_set:
                continue
                
            closed_set.add(state_str)
            
            for move in state.get_possible_moves():
                new_state = state.get_new_state(move)
                new_state_str = str(new_state.board)
                
                if new_state_str in closed_set:
                    continue
                
                new_state.h = new_state.calculate_misplaced_tiles()
                new_state.f = new_state.h  # Only heuristic matters for greedy
                
                heapq.heappush(open_set, new_state)
        
        end_time = time.time()
        return {'solution': None, 'time_taken': end_time - start_time}

# Function that serves as the bridge between GUI and solver
def solve_puzzle_algorithm(board, algorithm):
    solver = PuzzleSolver(board)
    
    if algorithm == 'BFS':
        return solver.bfs()
    elif algorithm == 'DFS':
        return solver.dfs()
    elif algorithm == 'IDS':
        return solver.ids()
    elif algorithm == 'Greedy':
        return solver.greedy()
    elif algorithm == 'A*':
        return solver.a_star()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# Add this class to create the GUI
class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")
        
        self.current_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solution = []
        self.solution_index = 0
        
        self.create_widgets()
        self.draw_board()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="8-Puzzle Solver", 
            font=("Montserrat", 24, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        title_label.pack()
        
        # Board frame
        self.board_frame = tk.Frame(
            self.root,
            width=300,
            height=300,
            bg="#ffffff",
            highlightbackground="#cccccc",
            highlightthickness=2
        )
        self.board_frame.pack(pady=20)
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Algorithm selection
        algo_frame = tk.Frame(control_frame, bg="#f0f0f0")
        algo_frame.pack(pady=10)
        
        algo_label = tk.Label(
            algo_frame, 
            text="Algorithm:", 
            font=("Inter", 12),
            bg="#f0f0f0"
        )
        algo_label.pack(side=tk.LEFT, padx=5)
        
        self.algo_var = tk.StringVar(value="A*")
        algo_dropdown = ttk.Combobox(
            algo_frame, 
            textvariable=self.algo_var,
            values=["BFS", "DFS", "IDS", "Greedy", "A*"],
            width=10,
            state="readonly"
        )
        algo_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        self.shuffle_button = tk.Button(
            button_frame,
            text="Shuffle",
            command=self.shuffle_board,
            font=("Inter", 11),
            bg="#67a1ff",
            fg="white",
            width=10,
            borderwidth=0,
            padx=10,
            pady=5
        )
        self.shuffle_button.pack(side=tk.LEFT, padx=5)
        
        self.solve_button = tk.Button(
            button_frame,
            text="Solve",
            command=self.solve_puzzle,
            font=("Inter", 11),
            bg="#67a1ff",
            fg="white",
            width=10,
            borderwidth=0,
            padx=10,
            pady=5
        )
        self.solve_button.pack(side=tk.LEFT, padx=5)
        
        self.step_button = tk.Button(
            button_frame,
            text="Step",
            command=self.step_solution,
            font=("Inter", 11),
            bg="#67a1ff",
            fg="white",
            width=10,
            borderwidth=0,
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset_board,
            font=("Inter", 11),
            bg="#67a1ff",
            fg="white",
            width=10,
            borderwidth=0,
            padx=10,
            pady=5
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Status and results
        self.status_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.status_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Ready",
            font=("Inter", 10),
            bg="#f0f0f0",
            fg="#333333",
            anchor=tk.W,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X)
        
        # Results text
        result_frame = tk.Frame(self.root, bg="#f0f0f0")
        result_frame.pack(pady=10, fill=tk.BOTH, padx=20, expand=True)
        
        self.result_text = tk.Text(
            result_frame,
            height=10,
            width=50,
            font=("Consolas", 10),
            bg="#ffffff",
            wrap=tk.WORD
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
    def draw_board(self):
        # Clear existing tiles
        for widget in self.board_frame.winfo_children():
            widget.destroy()
            
        # Create tiles
        tile_size = 90
        for i in range(3):
            for j in range(3):
                value = self.current_board[i][j]
                if value == 0:
                    tile = tk.Frame(
                        self.board_frame, 
                        width=tile_size, 
                        height=tile_size, 
                        bg="#f0f0f0"
                    )
                else:
                    tile = tk.Frame(
                        self.board_frame, 
                        width=tile_size, 
                        height=tile_size, 
                        bg="#67a1ff",
                        highlightbackground="#4d87e6",
                        highlightthickness=1
                    )
                    number = tk.Label(
                        tile, 
                        text=str(value), 
                        font=("Montserrat", 24, "bold"), 
                        bg="#67a1ff", 
                        fg="white"
                    )
                    number.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                    
                tile.place(x=j*tile_size + 15, y=i*tile_size + 15)
                
    def shuffle_board(self):
        # Reset status and solution
        self.solution = []
        self.solution_index = 0
        self.step_button.config(state=tk.DISABLED)
        self.status_label.config(text="Shuffling...")
        self.result_text.delete(1.0, tk.END)
        
        # Generate a random but solvable board
        while True:
            # Start with solved board
            board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
            # Apply random moves
            for _ in range(30):
                # Find the blank
                for i in range(3):
                    for j in range(3):
                        if board[i][j] == 0:
                            blank_i, blank_j = i, j
                
                # Get possible moves
                moves = []
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = blank_i + di, blank_j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        moves.append((ni, nj))
                
                # Apply a random move
                move_i, move_j = random.choice(moves)
                board[blank_i][blank_j], board[move_i][move_j] = board[move_i][move_j], board[blank_i][blank_j]
            
            # Check if board is solvable
            if self.is_solvable(board):
                self.current_board = board
                break
                
        self.draw_board()
        self.status_label.config(text="Board shuffled. Ready to solve.")
        
    def is_solvable(self, board):
        # Flatten the board
        flat_board = []
        for row in board:
            flat_board.extend(row)
            
        # Count inversions
        inversions = 0
        for i in range(len(flat_board)):
            if flat_board[i] == 0:
                continue
            for j in range(i + 1, len(flat_board)):
                if flat_board[j] == 0:
                    continue
                if flat_board[i] > flat_board[j]:
                    inversions += 1
                    
        # Find row of blank from bottom
        blank_row = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    blank_row = 3 - i
                    
        # Check if solvable
        if blank_row % 2 == 0:
            return inversions % 2 == 0
        else:
            return inversions % 2 == 1
            
    def solve_puzzle(self):
        self.result_text.delete(1.0, tk.END)
        algorithm = self.algo_var.get()
        self.status_label.config(text=f"Solving with {algorithm}...")
        self.root.update()
        
        # Solve the puzzle
        result = solve_puzzle_algorithm(self.current_board, algorithm)
        
        if result['solution']:
            self.solution = result['solution']
            self.solution_index = 0
            self.step_button.config(state=tk.NORMAL)
            
            # Display results
            self.result_text.insert(tk.END, f"{algorithm} Solution (Steps: {len(result['solution'])})\n")
            self.result_text.insert(tk.END, f"Moves: {result['solution']}\n")
            self.result_text.insert(tk.END, f"Time taken: {result['time_taken']:.5f} seconds\n")
            
            self.status_label.config(text=f"Solution found! {len(result['solution'])} steps, click 'Step' to visualize.")
        else:
            self.status_label.config(text=f"{algorithm} could not find a solution")
            messagebox.showerror("Error", "Could not find a solution for this puzzle.")
            
    def step_solution(self):
        if not self.solution or self.solution_index >= len(self.solution):
            self.step_button.config(state=tk.DISABLED)
            self.status_label.config(text="Solution complete!")
            return
            
        # Apply the next move
        move = self.solution[self.solution_index]
        self.solution_index += 1
        
        # Find the blank
        for i in range(3):
            for j in range(3):
                if self.current_board[i][j] == 0:
                    blank_i, blank_j = i, j
        
        # Apply the move
        if move == 'Up':
            self.current_board[blank_i][blank_j], self.current_board[blank_i-1][blank_j] = \
            self.current_board[blank_i-1][blank_j], self.current_board[blank_i][blank_j]
        elif move == 'Down':
            self.current_board[blank_i][blank_j], self.current_board[blank_i+1][blank_j] = \
            self.current_board[blank_i+1][blank_j], self.current_board[blank_i][blank_j]
        elif move == 'Left':
            self.current_board[blank_i][blank_j], self.current_board[blank_i][blank_j-1] = \
            self.current_board[blank_i][blank_j-1], self.current_board[blank_i][blank_j]
        elif move == 'Right':
            self.current_board[blank_i][blank_j], self.current_board[blank_i][blank_j+1] = \
            self.current_board[blank_i][blank_j+1], self.current_board[blank_i][blank_j]
            
        self.draw_board()
        self.status_label.config(text=f"Step {self.solution_index}/{len(self.solution)}: {move}")
        
    def reset_board(self):
        self.current_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solution = []
        self.solution_index = 0
        self.step_button.config(state=tk.DISABLED)
        self.status_label.config(text="Board reset to goal state.")
        self.result_text.delete(1.0, tk.END)
        self.draw_board()

# Modify the main function to run GUI when the program is executed
def main():
    # Check if we want to run the GUI or the console version
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--console':
        run_console_tests()
    else:
        root = tk.Tk()
        app = PuzzleGUI(root)
        root.mainloop()

def run_console_tests():
    # Original console code
    test_boards = [
        [[1, 2, 3], [4, 0, 6], [7, 5, 8]],
        [[1, 2, 3], [4, 5, 6], [0, 7, 8]],
        [[1, 2, 0], [4, 5, 3], [7, 8, 6]]
    ]

    algorithms = ['BFS', 'DFS', 'IDS', 'Greedy', 'A*']

    for board in test_boards:
        print(f"\nSolving Board: {board}")
        for algo in algorithms:
            result = solve_puzzle_algorithm(board, algo)
            if result['solution']:
                print(f"{algo} Solution (Steps: {len(result['solution'])})")
                print(f"Moves: {result['solution']}")
                print(f"Time taken: {result['time_taken']:.5f} seconds")
            else:
                print(f"{algo} could not find a solution")

if __name__ == "__main__":
    main()