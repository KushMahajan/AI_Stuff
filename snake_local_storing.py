import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Deep Q-Network Architecture
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Snake Game Environment
class SnakeGame:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = random.choice([(1,0), (0,1), (-1,0), (0,-1)])
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)
        
        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
        ]
        return np.array(state, dtype=int)
    
    def _is_collision(self, point):
        return (point in self.snake[1:] or
                point[0] < 0 or point[0] >= self.width or
                point[1] < 0 or point[1] >= self.height)
    
    def step(self, action):
        # Action: [straight, right, left]
        clock_wise = [(1,0), (0,1), (-1,0), (0,-1)]
        idx = clock_wise.index(self.direction)
        
        if action == 1:  # right turn
            new_idx = (idx + 1) % 4
        elif action == 2:  # left turn
            new_idx = (idx - 1) % 4
        else:  # straight
            new_idx = idx
            
        self.direction = clock_wise[new_idx]
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        self.steps += 1
        reward = 0
        game_over = False
        
        # Check collision
        if self._is_collision(new_head):
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over
        
        self.snake.insert(0, new_head)
        
        # Check food
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = -0.1  # Small negative reward for each step to encourage efficiency
        
        # Additional reward for moving towards food
        if abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1]) < \
           abs(head[0] - self.food[0]) + abs(head[1] - self.food[1]):
            reward += 0.1
            
        # Terminate if snake takes too many steps without eating
        if self.steps > 100 * len(self.snake):
            game_over = True
            
        return self._get_state(), reward, game_over

# AI Agent
class SnakeAI:
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_model = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=100000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.training_step = 0
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1
        
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, filename)
    
    def load(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=100000)

def plot_scores(scores, episodes):
    plt.figure(figsize=(8, 4))
    plt.plot(scores, color='blue')
    plt.title('Snake AI Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Convert plot to pygame surface
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    # Convert to pygame surface
    plot_surface = pygame.image.fromstring(raw_data, size, "RGB")
    plt.close()
    
    return plot_surface

def train_ai():
    pygame.init()
    cell_size = 20
    game = SnakeGame()
    
    # Create a larger window to accommodate the plot and score
    window_width = max(game.width * cell_size, 1920)  # At least 800px wide
    window_height = (game.height * cell_size) + 600  # Extra 300px for plot and score
    display = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Snake AI')
    clock = pygame.time.Clock()
    
    # Initialize font for score display
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)
    
    ai = SnakeAI()
    scores = []  # Keep track of scores for plotting
    
    # Try to load existing model
    if os.path.exists('snake_ai_model.pth'):
        ai.load('snake_ai_model.pth')
        print("Loaded existing model")
    
    episodes = 1000
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Clear display
            display.fill((0, 0, 0))
            
            # Draw game area
            game_surface = pygame.Surface((game.width * cell_size, game.height * cell_size))
            game_surface.fill((0, 0, 0))
            
            # Draw snake
            for segment in game.snake:
                pygame.draw.rect(game_surface, (0, 255, 0),
                               (segment[0] * cell_size, segment[1] * cell_size,
                                cell_size, cell_size))
            
            # Draw food
            pygame.draw.rect(game_surface, (255, 0, 0),
                           (game.food[0] * cell_size, game.food[1] * cell_size,
                            cell_size, cell_size))
            
            # Draw game surface centered horizontally
            game_x = (window_width - game.width * cell_size) // 2
            display.blit(game_surface, (game_x, 0))
            
            # Draw score and info
            score_text = font.render(f'Episode: {episode + 1}  Score: {game.score}  Best Score: {max(scores) if scores else 0}', 
                                   True, (255, 255, 255))
            display.blit(score_text, (10, game.height * cell_size + 10))
            
            # Draw epsilon value
            epsilon_text = font.render(f'Exploration Rate (ε): {ai.epsilon:.3f}', True, (255, 255, 255))
            display.blit(epsilon_text, (10, game.height * cell_size + 40))
            
            # Draw learning curve if we have scores
            if scores:
                plot = plot_scores(scores, episode)
                plot_x = (window_width - plot.get_width()) // 2
                display.blit(plot, (plot_x, game.height * cell_size + 70))
            
            pygame.display.flip()
            clock.tick(30)
            
            # AI action
            action = ai.act(state)
            next_state, reward, done = game.step(action)
            total_reward += reward
            
            ai.remember(state, action, reward, next_state, done)
            ai.train()
            
            state = next_state
            
            if done:
                scores.append(game.score)
                break
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            ai.save('snake_ai_model.pth')
            print(f"Episode {episode + 1}/{episodes}, Score: {game.score}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    train_ai()