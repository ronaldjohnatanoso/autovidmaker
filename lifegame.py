import pygame
import numpy as np
import sys

class GameOfLife:
    def __init__(self, width=800, height=600, cell_size=10):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Conway's Game of Life")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (255, 0, 255)
        self.CYAN = (0, 255, 255)
        
        # Initialize grid with random state
        self.grid = np.random.choice([0, 1], size=(self.grid_height, self.grid_width), p=[0.8, 0.2])
        
        self.running = True
        self.paused = False
        self.current_pattern = 0
        self.current_rule = 'conway'  # Default rule
        
    def get_rule_sets(self):
        """Return different cellular automaton rule sets"""
        return {
            'conway': 'Conway\'s Game of Life',
            'maze': 'Maze Generator (12345/3)',
            'coral': 'Coral Growth (45678/3)',
            'brain': 'Brian\'s Brain (Generations)',
            'seeds': 'Seeds (Explosive)',
            'highlife': 'HighLife (More stable)',
            'day_night': 'Day & Night (Symmetric)',
            'anneal': 'Anneal (Self-organizing)',
            'morley': 'Morley (Diamond patterns)',
            'life_without_death': 'Life Without Death'
        }
    
    def get_exciting_patterns(self):
        """Return dictionary of exciting Conway's Game of Life patterns"""
        patterns = {
            'glider_gun': [
                ".........................",
                ".......................X.",
                "...................X.XX.",
                "..............X.X......XX",
                ".....X.X.....X...........XX",
                ".....X.X.....X.X.........X.X",
                "..............X...........X",
                "...................X.XX..",
                ".......................X.",
                "........................."
            ],
            'penta_decathlon': [
                "..XXX..",
                "..X.X..",
                "..X.X..",
                "..XXX..",
                ".......",
                "..XXX..",
                "..X.X..",
                "..X.X..",
                "..XXX.."
            ],
            'pulsar': [
                "..XXX...XXX..",
                ".............",
                "X....X.X....X",
                "X....X.X....X",
                "X....X.X....X",
                "..XXX...XXX..",
                ".............",
                "..XXX...XXX..",
                "X....X.X....X",
                "X....X.X....X",
                "X....X.X....X",
                ".............",
                "..XXX...XXX.."
            ],
            'r_pentomino': [
                ".XX",
                "XX.",
                ".X."
            ],
            'acorn': [
                ".X.....",
                "...X...",
                "XX..XXX"
            ],
            'diehard': [
                "......X.",
                "XX......",
                ".X...XXX"
            ],
            'queen_bee_shuttle': [
                "X.........X",
                "XXX.....XXX",
                "...XX.XX...",
                "..X.X.X.X..",
                "X.X.....X.X",
                "X.X.....X.X",
                "XX.......XX"
            ],
            'glider': [
                ".X.",
                "..X",
                "XXX"
            ]
        }
        return patterns
    
    def update_grid(self):
        """Apply cellular automaton rules based on current rule set"""
        neighbors = self.count_neighbors(self.grid)
        
        if self.current_rule == 'conway':
            # Standard Conway's Game of Life: B3/S23
            new_grid = np.zeros_like(self.grid)
            new_grid[(self.grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
            new_grid[(self.grid == 0) & (neighbors == 3)] = 1
            
        elif self.current_rule == 'maze':
            # Maze: B3/S12345 - Creates maze-like patterns
            new_grid = np.zeros_like(self.grid)
            survive_conditions = (neighbors >= 1) & (neighbors <= 5)
            new_grid[(self.grid == 1) & survive_conditions] = 1
            new_grid[(self.grid == 0) & (neighbors == 3)] = 1
            
        elif self.current_rule == 'coral':
            # Coral: B3/S45678 - Creates coral-like growth
            new_grid = np.zeros_like(self.grid)
            survive_conditions = (neighbors >= 4) & (neighbors <= 8)
            new_grid[(self.grid == 1) & survive_conditions] = 1
            new_grid[(self.grid == 0) & (neighbors == 3)] = 1
            
        elif self.current_rule == 'seeds':
            # Seeds: B2/S - Explosive growth, cells die after one generation
            new_grid = np.zeros_like(self.grid)
            new_grid[(self.grid == 0) & (neighbors == 2)] = 1
            
        elif self.current_rule == 'highlife':
            # HighLife: B36/S23 - Like Conway but with more births
            new_grid = np.zeros_like(self.grid)
            new_grid[(self.grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
            birth_conditions = (neighbors == 3) | (neighbors == 6)
            new_grid[(self.grid == 0) & birth_conditions] = 1
            
        elif self.current_rule == 'day_night':
            # Day & Night: B3678/S34678 - Symmetric rule
            new_grid = np.zeros_like(self.grid)
            survive_conditions = ((neighbors == 3) | (neighbors == 4) | 
                                (neighbors == 6) | (neighbors == 7) | (neighbors == 8))
            new_grid[(self.grid == 1) & survive_conditions] = 1
            birth_conditions = ((neighbors == 3) | (neighbors == 6) | 
                              (neighbors == 7) | (neighbors == 8))
            new_grid[(self.grid == 0) & birth_conditions] = 1
            
        elif self.current_rule == 'anneal':
            # Anneal: B4678/S35678 - Self-organizing
            new_grid = np.zeros_like(self.grid)
            survive_conditions = ((neighbors == 3) | (neighbors == 5) | 
                                (neighbors == 6) | (neighbors == 7) | (neighbors == 8))
            new_grid[(self.grid == 1) & survive_conditions] = 1
            birth_conditions = ((neighbors == 4) | (neighbors == 6) | 
                              (neighbors == 7) | (neighbors == 8))
            new_grid[(self.grid == 0) & birth_conditions] = 1
            
        elif self.current_rule == 'morley':
            # Morley: B368/S245 - Creates diamond patterns
            new_grid = np.zeros_like(self.grid)
            survive_conditions = ((neighbors == 2) | (neighbors == 4) | (neighbors == 5))
            new_grid[(self.grid == 1) & survive_conditions] = 1
            birth_conditions = ((neighbors == 3) | (neighbors == 6) | (neighbors == 8))
            new_grid[(self.grid == 0) & birth_conditions] = 1
            
        elif self.current_rule == 'life_without_death':
            # Life Without Death: B3/S012345678 - Cells never die
            new_grid = np.copy(self.grid)  # Keep all living cells
            new_grid[(self.grid == 0) & (neighbors == 3)] = 1
            
        elif self.current_rule == 'brain':
            # Brian's Brain - 3 states: 0=dead, 1=alive, 2=dying
            # This needs special handling for 3 states
            if not hasattr(self, 'brain_grid'):
                self.brain_grid = np.zeros_like(self.grid)
            
            new_grid = np.zeros_like(self.grid)
            # Count only "alive" neighbors (state 1)
            alive_neighbors = self.count_neighbors(self.grid == 1)
            
            # Dead cells with exactly 2 alive neighbors become alive
            new_grid[(self.grid == 0) & (alive_neighbors == 2)] = 1
            
            # Store previous state for drawing dying cells
            self.brain_grid = np.copy(self.grid)
        
        else:
            # Fallback to Conway's rules
            new_grid = np.zeros_like(self.grid)
            new_grid[(self.grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
            new_grid[(self.grid == 0) & (neighbors == 3)] = 1
        
        self.grid = new_grid
    
    def draw_grid(self):
        """Draw the current state of the grid with rule-specific colors"""
        self.screen.fill(self.BLACK)
        
        # Choose colors based on current rule
        if self.current_rule == 'maze':
            live_color = self.YELLOW
        elif self.current_rule == 'coral':
            live_color = self.CYAN
        elif self.current_rule == 'seeds':
            live_color = self.RED
        elif self.current_rule == 'highlife':
            live_color = self.GREEN
        elif self.current_rule == 'day_night':
            live_color = self.BLUE
        elif self.current_rule == 'brain':
            live_color = self.WHITE
            dying_color = self.RED
        else:
            live_color = self.WHITE
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                if self.current_rule == 'brain' and hasattr(self, 'brain_grid'):
                    # Special drawing for Brian's Brain
                    if self.grid[y, x] == 1:
                        color = live_color
                    elif self.brain_grid[y, x] == 1 and self.grid[y, x] == 0:
                        color = dying_color
                    else:
                        color = self.BLACK
                else:
                    color = live_color if self.grid[y, x] == 1 else self.BLACK
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)
    
    def count_neighbors(self, grid):
        """Count living neighbors for each cell using numpy convolution"""
        # Create kernel for counting neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Pad the grid to handle edges
        padded_grid = np.pad(grid, 1, mode='wrap')
        
        # Count neighbors using convolution
        neighbor_count = np.zeros_like(grid)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:  # Skip center cell
                    continue
                neighbor_count += padded_grid[i:i+grid.shape[0], j:j+grid.shape[1]]
        
        return neighbor_count
    
    def handle_mouse_click(self, pos):
        """Toggle cell state when clicked"""
        x, y = pos
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid[grid_y, grid_x] = 1 - self.grid[grid_y, grid_x]
    
    def load_pattern(self, pattern_name, start_x=None, start_y=None):
        """Load a pattern onto the grid"""
        patterns = self.get_exciting_patterns()
        if pattern_name not in patterns:
            return
            
        pattern = patterns[pattern_name]
        
        # Clear grid first
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Calculate center position if not specified
        if start_x is None:
            start_x = (self.grid_width - len(pattern[0])) // 2
        if start_y is None:
            start_y = (self.grid_height - len(pattern)) // 2
            
        # Place pattern on grid
        for y, row in enumerate(pattern):
            for x, cell in enumerate(row):
                grid_x = start_x + x
                grid_y = start_y + y
                
                if (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height and 
                    cell == 'X'):
                    self.grid[grid_y, grid_x] = 1
    
    def load_multiple_gliders(self):
        """Load multiple gliders moving in different directions"""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Glider patterns for different directions
        gliders = {
            'se': [["..X"], [".XX"], ["XX."]],  # Southeast
            'sw': [["X.."], ["XX."], [".XX"]],  # Southwest  
            'ne': [["XX."], [".XX"], ["..X"]],  # Northeast
            'nw': [[".XX"], ["XX."], ["X.."]]   # Northwest
        }
        
        # Place gliders in corners
        positions = [
            ('se', 5, 5),
            ('sw', self.grid_width-8, 5),
            ('ne', 5, self.grid_height-8),
            ('nw', self.grid_width-8, self.grid_height-8)
        ]
        
        for direction, x, y in positions:
            pattern = gliders[direction]
            for py, row in enumerate(pattern):
                for px, cell in enumerate(row[0]):
                    if cell == 'X' and 0 <= x+px < self.grid_width and 0 <= y+py < self.grid_height:
                        self.grid[y+py, x+px] = 1
    
    def run(self):
        """Main game loop"""
        pattern_names = list(self.get_exciting_patterns().keys())
        rule_names = list(self.get_rule_sets().keys())
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        # Reset with random state
                        self.grid = np.random.choice([0, 1], 
                                                   size=(self.grid_height, self.grid_width), 
                                                   p=[0.8, 0.2])
                    elif event.key == pygame.K_c:
                        # Clear grid
                        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
                    elif event.key == pygame.K_n:
                        # Next exciting pattern
                        self.current_pattern = (self.current_pattern + 1) % len(pattern_names)
                        self.load_pattern(pattern_names[self.current_pattern])
                    elif event.key == pygame.K_g:
                        # Load multiple gliders
                        self.load_multiple_gliders()
                    elif event.key == pygame.K_TAB:
                        # Cycle through rule sets
                        current_idx = rule_names.index(self.current_rule)
                        self.current_rule = rule_names[(current_idx + 1) % len(rule_names)]
                        pygame.display.set_caption(f"Cellular Automaton - {self.get_rule_sets()[self.current_rule]}")
                    elif event.key == pygame.K_1:
                        self.load_pattern('glider_gun')
                    elif event.key == pygame.K_2:
                        self.load_pattern('pulsar')
                    elif event.key == pygame.K_3:
                        self.load_pattern('r_pentomino')
                    elif event.key == pygame.K_4:
                        self.load_pattern('acorn')
                    elif event.key == pygame.K_5:
                        self.load_pattern('diehard')
                        
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_mouse_click(event.pos)
            
            # Update grid if not paused
            if not self.paused:
                self.update_grid()
            
            # Draw everything
            self.draw_grid()
            
            # Display controls
            font = pygame.font.Font(None, 20)
            controls = [
                "SPACE: Pause/Resume  |  R: Random  |  C: Clear  |  TAB: Change Rules",
                "N: Next Pattern  |  G: Multiple Gliders  |  Click: Toggle cell",
                "1: Glider Gun  |  2: Pulsar  |  3: R-Pentomino  |  4: Acorn  |  5: Diehard"
            ]
            
            for i, text in enumerate(controls):
                rendered = font.render(text, True, self.WHITE)
                self.screen.blit(rendered, (10, 10 + i * 18))
            
            # Show current rule and pattern
            rule_text = f"Rule: {self.get_rule_sets()[self.current_rule]}"
            rendered = font.render(rule_text, True, (255, 255, 0))
            self.screen.blit(rendered, (10, self.height - 50))
            
            if hasattr(self, 'current_pattern'):
                pattern_text = f"Pattern: {pattern_names[self.current_pattern]}"
                rendered = font.render(pattern_text, True, (0, 255, 0))
                self.screen.blit(rendered, (10, self.height - 30))
            
            if self.paused:
                paused_text = pygame.font.Font(None, 36).render("PAUSED", True, (255, 0, 0))
                self.screen.blit(paused_text, (self.width - 100, 10))
            
            pygame.display.flip()
            self.clock.tick(20)  # Increased to 20 FPS for more excitement
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameOfLife(width=1000, height=800, cell_size=8)
    game.run()