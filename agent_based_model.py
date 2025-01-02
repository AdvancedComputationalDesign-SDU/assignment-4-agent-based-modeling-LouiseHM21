import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Set a seed for reproducibility
random.seed(5)  # Seed for Python's random module
np.random.seed(5)  # Seed for NumPy's random module

# Define the environment
class Environment:
    """
    Represents the simulation environment, including boundaries, obstacles, and goals.
    """
    def __init__(self, width, height, obstacles, goals):
        self.width = width
        self.height = height
        self.obstacles = obstacles  # List of obstacles represented as rectangles [(x1, y1, x2, y2)]
        self.goals = goals  # List of goal positions [(x, y)]

    def is_within_bounds(self, position):
        """
        Check if a position is within the environment boundaries.
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, position):
        """
        Check if a position is inside any obstacle.
        """
        for obstacle in self.obstacles:
            x1, y1, x2, y2 = obstacle
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                return True
        return False

    def is_valid_position(self, position):
        """
        Check if a position is valid (inside boundaries and not inside obstacles).
        """
        return self.is_within_bounds(position) and not self.is_obstacle(position)

# Define the agent
class Agent:
    """
    Represents an agent with behaviors such as goal-seeking and obstacle avoidance.
    """
    def __init__(self, position, goal, speed=0.5, personal_space=1.5, wanderer=False):
        self.position = np.array(position, dtype=float)  # Current position of the agent
        self.path = [np.array(position, dtype=float)]  # Path to visualize movement
        self.goal = goal  # Goal assigned to the agent
        self.speed = speed  # Maximum movement speed
        self.personal_space = personal_space  # Distance to maintain from other agents
        self.velocity = np.array([0.0, 0.0])  # Current velocity
        self.wanderer = wanderer  # Flag for wanderer behavior (random movement)

    def compute_velocity(self, neighbors, environment):
        """
        Compute the agent's velocity based on its goal and nearby obstacles.
        """
        self.velocity = np.array([0.0, 0.0])  # Reset velocity

        if self.wanderer:
            # Random movement for wanderer agents
            self.velocity = np.random.uniform(-1, 1, size=2)
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
            return

        if self.goal is not None:
            # Goal-seeking behavior
            direction_to_goal = self.goal - self.position
            if np.linalg.norm(direction_to_goal) > 0:
                direction_to_goal /= np.linalg.norm(direction_to_goal)  # Normalize
            self.velocity += direction_to_goal * 0.5  # Weight for goal-seeking

        # Normalize and limit to the agent's speed
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

    def move(self, environment):
        """
        Update the agent's position based on its velocity, ensuring it stays in valid areas.
        If stuck, assign a new random goal.
        """
        new_position = self.position + self.velocity
        if environment.is_valid_position(new_position):
            self.position = new_position
            self.path.append(self.position.copy())  # Update the path with the new position
        else:
            # If the movement leads to an invalid position, assign a new goal
            self.goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])
            while not environment.is_valid_position(self.goal):
                self.goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])

# Utility functions
def create_obstacles(width, height, num_obstacles=15):
    """
    Generate a list of non-overlapping rectangular obstacles.
    """
    obstacles = []
    while len(obstacles) < num_obstacles:
        x1 = random.uniform(0, width - 2)
        y1 = random.uniform(0, height - 2)
        x2 = x1 + random.uniform(0.5, 2)  # Random width
        y2 = y1 + random.uniform(0.5, 2)  # Random height

        overlap = any(not (x2 <= obs[0] or x1 >= obs[2] or y2 <= obs[1] or y1 >= obs[3]) for obs in obstacles)
        if not overlap:
            obstacles.append((x1, y1, x2, y2))
    return obstacles

def create_goals(environment):
    """
    Generate valid goal positions in the environment (not inside obstacles).
    """
    goals = []
    for _ in range(5):  # Generate 5 goals
        valid_goal = False
        while not valid_goal:
            goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])
            valid_goal = environment.is_valid_position(goal)
        goals.append(goal)
    return goals

def initialize_simulation(num_agents, num_wanderers, env_width, env_height, obstacles, spawn_points):
    """
    Initialize the environment and agents for the simulation.
    """
    environment = Environment(
        width=env_width,
        height=env_height,
        obstacles=obstacles,
        goals=[]
    )
    goals = create_goals(environment)
    environment.goals = goals

    agents = []
    goal_index = 0
    for _ in range(num_agents):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        while not environment.is_valid_position(start_pos):
            start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, goals[goal_index]))
        goal_index = (goal_index + 1) % len(goals)

    for _ in range(num_wanderers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        while not environment.is_valid_position(start_pos):
            start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, None, wanderer=True))

    return environment, agents

def run_simulation(environment, agents, spawn_points, steps=100, save_dir="simulation_images"):
    """
    Run the simulation for a specified number of steps, visualizing the results.
    Saves images at each step if a save directory is specified.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(15, 8))  # Increased figure size to accommodate the legend

    # Legend handles for obstacles, agents, goals, and spawn points
    obstacles_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Obstacle')
    agents_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Agent')
    wanderers_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Wanderer')  # Smaller size for wanderers
    goals_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Goal')
    spawn_patch = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='Spawn Point')

    for step in range(steps):
        plt.clf()
        plt.subplots_adjust(right=0.8)  # Adjust right margin to make space for the legend
        plt.xlim(0, environment.width)
        plt.ylim(0, environment.height)

        for obstacle in environment.obstacles:
            x_vals = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
            y_vals = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
            plt.plot(x_vals, y_vals, color='black', lw=2)

        for goal in environment.goals:
            plt.scatter(goal[0], goal[1], color='red', s=100, marker='o')

        for spawn in spawn_points:
            plt.scatter(spawn[0], spawn[1], color='purple', s=100, marker='D')  # Visualize spawn points

        for agent in agents:
            neighbors = [a for a in agents if a != agent and np.linalg.norm(a.position - agent.position) < agent.personal_space]
            agent.compute_velocity(neighbors, environment)
            agent.move(environment)

            # Draw the complete movement trail
            path = np.array(agent.path)
            plt.plot(path[:, 0], path[:, 1], color='gray', linestyle='--', linewidth=0.5)

            # Different sizes for wanderers
            size = 20 if agent.wanderer else 50
            plt.scatter(agent.position[0], agent.position[1], color='blue', s=size)

        # Randomly spawn new agents
        if step % 10 == 0:  # Spawn new agents every 10 steps
            for _ in range(3):
                spawn_point = random.choice(spawn_points)
                while not environment.is_valid_position(spawn_point):
                    spawn_point = random.choice(spawn_points)
                new_goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])
                while not environment.is_valid_position(new_goal):
                    new_goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])
                agents.append(Agent(spawn_point, new_goal))

        # Add the legend after all visual elements are drawn
        plt.legend(handles=[obstacles_patch, agents_patch, wanderers_patch, goals_patch, spawn_patch], loc='center left', bbox_to_anchor=(1.05, 0.5))

        plt.xticks([])
        plt.yticks([])

        # Save the frame as an image
        frame_filename = os.path.join(save_dir, f"frame_{step:03d}.png")
        plt.savefig(frame_filename, bbox_inches='tight')

        plt.pause(0.1)

    plt.show()

# Main execution
if __name__ == "__main__":
    env_width, env_height = 20, 20

    # Define or generate obstacles
    obstacles = create_obstacles(env_width, env_height, num_obstacles=15)

    # Define spawn points
    spawn_points = [(2, 2), (18, 18), (15, 3)]

    # Create the environment and initialize agents
    env, agents = initialize_simulation(num_agents=50, num_wanderers=20,
                                        env_width=env_width, env_height=env_height, obstacles=obstacles, spawn_points=spawn_points)

    # Run the simulation
    run_simulation(env, agents, spawn_points, steps=100, save_dir="C:\\Users\\Louis\\OneDrive\\7th semester\\Advanced Computational Design\\assignment-4-agent-based-modeling-LouiseHM21\\simulation_images1")
