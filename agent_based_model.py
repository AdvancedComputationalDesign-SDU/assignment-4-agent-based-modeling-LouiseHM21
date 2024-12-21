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

    def is_goal_valid(self, goal_position):
        """
        Ensure that a goal position is not inside any obstacle.
        """
        return not self.is_obstacle(goal_position)

# Define the agent
class Agent:
    """
    Represents an agent with behaviors such as goal-seeking, obstacle avoidance, 
    and optional wanderer or group-seeking dynamics.
    """
    def __init__(self, position, goal, speed=0.5, personal_space=1.5, wanderer=False, group_seeker=False, path_follower=False):
        self.position = np.array(position, dtype=float)  # Current position of the agent
        self.goal = goal  # Goal assigned to the agent
        self.speed = speed  # Maximum movement speed
        self.personal_space = personal_space  # Distance to maintain from other agents
        self.velocity = np.array([0.0, 0.0])  # Current velocity
        self.wanderer = wanderer  # Flag for wanderer behavior (random movement)
        self.group_seeker = group_seeker  # Flag for group-seeking behavior
        self.path_follower = path_follower  # Flag for path-following behavior

    def compute_velocity(self, neighbors, environment):
        """
        Compute the agent's velocity based on its goal, nearby obstacles, and neighbors.
        """
        # Step 1: Avoid obstacles (stronger priority)
        self.avoid_obstacles(environment)

        if self.wanderer:
            # Random movement for agents marked as wanderers
            self.velocity = np.random.uniform(-1, 1, size=2)
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
            return

        if self.group_seeker:
            # Move towards the center of nearby group members
            group_center = np.array([0.0, 0.0])
            num_neighbors = 0
            for neighbor in neighbors:
                if np.linalg.norm(self.position - neighbor.position) < 15:  # Extended radius for group-seeking behavior
                    group_center += neighbor.position
                    num_neighbors += 1

            if num_neighbors > 0:
                group_center /= num_neighbors
                direction_to_group = group_center - self.position
                distance_to_group = np.linalg.norm(direction_to_group)

                if distance_to_group > 0:
                    direction_to_group /= distance_to_group  # Normalize direction vector
                    self.velocity += direction_to_group * 1.0  # Increased attraction to group center

        if self.goal is not None:
            # Goal-seeking behavior
            direction_to_goal = self.goal - self.position
            distance_to_goal = np.linalg.norm(direction_to_goal)

            if distance_to_goal > 0:
                direction_to_goal /= distance_to_goal  # Normalize direction vector
            else:
                direction_to_goal = np.array([0.0, 0.0])  # No movement if already at goal

            self.velocity += direction_to_goal * 0.5  # Apply goal-seeking force with reduced weight

        # Normalize velocity to match the agent's speed
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

    def avoid_obstacles(self, environment):
        """
        Detect obstacles in the agent's path and apply a repulsion force to steer away.
        """
        # Predict the next step based on current velocity
        step_ahead = self.position + self.velocity * 1.0

        for obstacle in environment.obstacles:
            x1, y1, x2, y2 = obstacle
            if x1 <= step_ahead[0] <= x2 and y1 <= step_ahead[1] <= y2:
                # Adjust velocity to avoid obstacle
                self.steer_around_obstacle(obstacle)
                break

    def steer_around_obstacle(self, obstacle):
        """
        Adjust velocity to smoothly navigate around an obstacle.
        """
        # Calculate the center of the obstacle
        x1, y1, x2, y2 = obstacle
        obstacle_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Steer away from the obstacle center
        avoidance_direction = self.position - obstacle_center
        avoidance_direction /= np.linalg.norm(avoidance_direction)  # Normalize direction

        # Rotate the velocity slightly for smoother avoidance
        angle_change = np.pi / 4  # Rotate 45 degrees
        rotation_matrix = np.array([[np.cos(angle_change), -np.sin(angle_change)],
                                    [np.sin(angle_change), np.cos(angle_change)]])
        self.velocity = np.dot(rotation_matrix, self.velocity)

    def move(self):
        """
        Update the agent's position based on its velocity.
        """
        self.position += self.velocity

    def reached_goal(self):
        """
        Check if the agent has reached its assigned goal.
        """
        if self.goal is not None:
            distance_to_goal = np.linalg.norm(self.position - self.goal)
            if distance_to_goal < 0.5:
                self.goal = None  # Goal reached
        return self.goal is None


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
            valid_goal = environment.is_goal_valid(goal)
        goals.append(goal)
    return goals


def initialize_simulation(num_agents, num_wanderers, num_group_seekers, num_path_followers, env_width, env_height):
    """
    Initialize the environment and agents for the simulation.
    """
    environment = Environment(
        width=env_width,
        height=env_height,
        obstacles=create_obstacles(env_width, env_height),
        goals=[]
    )
    goals = create_goals(environment)
    environment.goals = goals

    agents = []
    goal_index = 0
    for _ in range(num_agents):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, goals[goal_index]))
        goal_index = (goal_index + 1) % len(goals)

    for _ in range(num_wanderers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, None, wanderer=True))

    for _ in range(num_group_seekers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, None, group_seeker=True))

    for _ in range(num_path_followers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        agents.append(Agent(start_pos, None, path_follower=True))

    return environment, agents


def add_new_agents(spawn_points, agents, num_new_agents, env_width, env_height):
    """
    Add new agents to the simulation at specified spawn points.
    """
    for _ in range(num_new_agents):
        spawn_point = random.choice(spawn_points)
        goal = np.array([random.uniform(0, env_width), random.uniform(0, env_height)])
        agents.append(Agent(spawn_point, goal))


def run_simulation(environment, agents, spawn_points, steps=100, save_dir="simulation_images"):
    """
    Run the simulation for a specified number of steps, visualizing the results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 8))
    for step in range(steps):
        plt.clf()
        plt.xlim(0, environment.width)
        plt.ylim(0, environment.height)

        # Draw obstacles
        for obstacle in environment.obstacles:
            x_vals = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
            y_vals = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
            plt.plot(x_vals, y_vals, color='black')

        # Draw goals
        for goal in environment.goals:
            plt.scatter(goal[0], goal[1], color='red')

        # Add new agents at spawn points every 10 steps
        if step % 10 == 0:
            add_new_agents(spawn_points, agents, num_new_agents=3, env_width=environment.width, env_height=environment.height)

        # Update and draw agents
        for agent in agents:
            neighbors = [a for a in agents if a != agent and np.linalg.norm(a.position - agent.position) < agent.personal_space]
            agent.compute_velocity(neighbors, environment)
            agent.move()
            plt.scatter(agent.position[0], agent.position[1], color='blue')

        plt.pause(0.1)

    plt.show()

# Main execution
if __name__ == "__main__":
    env_width, env_height = 20, 20
    spawn_points = [(2, 2), (18, 18), (15, 3)]
    env, agents = initialize_simulation(num_agents=50, num_wanderers=20, num_group_seekers=25, num_path_followers=25, env_width=env_width, env_height=env_height)
    run_simulation(env, agents, spawn_points)
