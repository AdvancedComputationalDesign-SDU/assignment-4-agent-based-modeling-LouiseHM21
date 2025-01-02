import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Set a seed for reproducibility to ensure consistent simulation results
random.seed(5)  # Seed for Python's random module
np.random.seed(5)  # Seed for NumPy's random module

# Define the environment
class Environment:
    """
    Represents the simulation environment, including boundaries, obstacles, and goals.
    """
    def __init__(self, width, height, obstacles, goals):
        self.width = width  # Width of the environment
        self.height = height  # Height of the environment
        self.obstacles = obstacles  # List of rectangular obstacles represented as [(x1, y1, x2, y2)]
        self.goals = goals  # List of goal positions represented as [(x, y)]

    def is_within_bounds(self, position):
        """
        Check if a given position is within the boundaries of the environment.
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, position):
        """
        Check if a given position lies inside any obstacle.
        """
        for obstacle in self.obstacles:
            x1, y1, x2, y2 = obstacle
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                return True
        return False

    def is_valid_position(self, position):
        """
        Check if a position is valid (within boundaries and not inside obstacles).
        """
        return self.is_within_bounds(position) and not self.is_obstacle(position)

# Define the agent
class Agent:
    """
    Represents an agent with behaviors such as goal-seeking and obstacle avoidance.
    """
    def __init__(self, position, goal, speed=0.5, personal_space=1.5, wanderer=False):
        self.position = np.array(position, dtype=float)  # Current position of the agent
        self.path = [np.array(position, dtype=float)]  # Stores the path taken by the agent
        self.goal = goal  # Goal position assigned to the agent
        self.speed = speed  # Maximum movement speed of the agent
        self.personal_space = personal_space  # Minimum distance to maintain from other agents
        self.velocity = np.array([0.0, 0.0])  # Initial velocity of the agent
        self.wanderer = wanderer  # Determines if the agent moves randomly

    def compute_velocity(self, neighbors, environment):
        """
        Compute the agent's velocity based on its goal and nearby obstacles or agents.
        """
        self.velocity = np.array([0.0, 0.0])  # Reset velocity

        if self.wanderer:
            # If the agent is a wanderer, assign random velocity.
            self.velocity = np.random.uniform(-1, 1, size=2)
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
            return

        if self.goal is not None:
            # Calculate direction to the goal.
            direction_to_goal = self.goal - self.position
            if np.linalg.norm(direction_to_goal) > 0:
                direction_to_goal /= np.linalg.norm(direction_to_goal)  # Normalize the vector
            self.velocity += direction_to_goal * 0.5  # Weighted contribution towards goal

        # Normalize velocity to maintain maximum allowed speed
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

    def move(self, environment):
        """
        Update the agent's position based on its velocity, ensuring valid movement.
        Assign a new random goal if the agent encounters an invalid position.
        """
        new_position = self.position + self.velocity
        if environment.is_valid_position(new_position):
            self.position = new_position
            self.path.append(self.position.copy())  # Record the new position in the agent's path
        else:
            # If the position is invalid, reassign a random valid goal within the environment
            self.goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])
            while not environment.is_valid_position(self.goal):
                self.goal = np.array([random.uniform(1, environment.width - 1), random.uniform(1, environment.height - 1)])

# Utility functions

def create_obstacles(width, height, num_obstacles=15):
    """
    Generate a list of non-overlapping rectangular obstacles within the environment.
    """
    obstacles = []
    while len(obstacles) < num_obstacles:
        x1 = random.uniform(0, width - 2)
        y1 = random.uniform(0, height - 2)
        x2 = x1 + random.uniform(0.5, 2)  # Random width of the obstacle
        y2 = y1 + random.uniform(0.5, 2)  # Random height of the obstacle

        # Check for overlap with existing obstacles
        overlap = any(not (x2 <= obs[0] or x1 >= obs[2] or y2 <= obs[1] or y1 >= obs[3]) for obs in obstacles)
        if not overlap:
            obstacles.append((x1, y1, x2, y2))
    return obstacles

def create_goals(environment):
    """
    Generate a list of valid goal positions within the environment, avoiding obstacles.
    """
    goals = []
    for _ in range(10):  # Generate 10 goals
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
    goals = create_goals(environment)  # Create goals within the environment
    environment.goals = goals

    agents = []
    goal_index = 0
    for _ in range(num_agents):
        # Randomly assign starting positions for agents
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        while not environment.is_valid_position(start_pos):
            start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, goals[goal_index]))
        goal_index = (goal_index + 1) % len(goals)

    for _ in range(num_wanderers):
        # Randomly assign starting positions for wanderer agents
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        while not environment.is_valid_position(start_pos):
            start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, None, wanderer=True))

    return environment, agents

def run_simulation(environment, agents, spawn_points, steps=50, save_dir="simulation_images"):
    """
    Run the simulation for a specified number of steps, visualizing the results.
    Optionally saves images of each step to the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory to save simulation images if it doesn't exist

    plt.figure(figsize=(15, 8))  # Create a figure for visualization

    for step in range(steps):
        plt.clf()  # Clear the figure for the next frame
        plt.xlim(0, environment.width)  # Set x-axis limits
        plt.ylim(0, environment.height)  # Set y-axis limits

        for obstacle in environment.obstacles:
            # Draw each obstacle as a rectangle
            x_vals = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
            y_vals = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
            plt.plot(x_vals, y_vals, color='black', lw=2)

        for goal in environment.goals:
            # Plot each goal as a red circle
            plt.scatter(goal[0], goal[1], color='red', s=100, marker='o')

        for spawn in spawn_points:
            # Plot each spawn point as a purple diamond
            plt.scatter(spawn[0], spawn[1], color='purple', s=100, marker='D')

        for agent in agents:
            # Compute and update agent's movement
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
        # frame_filename = os.path.join(save_dir, f"frame_{step:03d}.png")
        # plt.savefig(frame_filename, bbox_inches='tight')

        plt.pause(0.1)

    plt.show()

# Main execution
if __name__ == "__main__":
    env_width, env_height = 20, 20

    # Define or generate obstacles
    obstacles = create_obstacles(env_width, env_height, num_obstacles=30)

    # Define spawn points
    spawn_points = [(2, 2), (18, 18), (15, 3), (10, 1)]

    # Create the environment and initialize agents
    env, agents = initialize_simulation(num_agents=75, num_wanderers=20,
                                        env_width=env_width, env_height=env_height, obstacles=obstacles, spawn_points=spawn_points)

    # Run the simulation
    # run_simulation(env, agents, spawn_points, steps=50, save_dir="C:\\Users\\Louis\\OneDrive\\7th semester\\Advanced Computational Design\\assignment-4-agent-based-modeling-LouiseHM21\\simulation_images3")
    run_simulation(env, agents, spawn_points)