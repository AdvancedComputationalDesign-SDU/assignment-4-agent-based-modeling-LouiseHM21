import numpy as np
import matplotlib.pyplot as plt
import random

# Set a seed for reproducibility
random.seed(5)  # Seed for Python's random module
np.random.seed(5)  # Seed for NumPy's random module

# Define the environment
class Environment:
    def __init__(self, width, height, obstacles, goals):
        self.width = width
        self.height = height
        self.obstacles = obstacles  # List of obstacle outlines [(x1, y1), (x2, y2), ...]
        self.goals = goals  # Fixed list of goals [(x1, y1), (x2, y2), ...]

    def is_within_bounds(self, position):
        """Check if the position is inside the environment."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, position):
        """Check if the position is inside any obstacle."""
        for obstacle in self.obstacles:
            x1, y1, x2, y2 = obstacle
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                return True
        return False

    def is_goal_valid(self, goal_position):
        """Check if the goal is outside any obstacle."""
        return not self.is_obstacle(goal_position)

# Define the agent
class Agent:
    def __init__(self, position, goal, speed=0.5, personal_space=1.5, wanderer=False, group_seeker=False, path_follower=False):
        self.position = np.array(position, dtype=float)  # Current position as a numpy array
        self.goal = goal  # Single goal assigned from the fixed list of goals
        self.speed = speed                               # Maximum movement speed
        self.personal_space = personal_space             # Distance to maintain from other agents
        self.velocity = np.array([0.0, 0.0])             # Current velocity
        self.wanderer = wanderer                         # Flag to identify wanderers
        self.group_seeker = group_seeker                 # Flag for group-seeking behavior
        self.path_follower = path_follower               # Flag for path-following behavior

    def compute_velocity(self, neighbors, environment):
        """Compute the agent's velocity based on goal, obstacles, and neighbors."""
        
        # Avoid obstacles first (stronger repulsion)
        self.avoid_obstacles(environment)

        if self.wanderer:
            # Random movement for wanderers
            self.velocity = np.random.uniform(-1, 1, size=2)
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
            return

        if self.group_seeker:
            # Group-seeking behavior (move towards the center of the nearby group)
            group_center = np.array([0.0, 0.0])
            num_neighbors = 0
            for neighbor in neighbors:
                if np.linalg.norm(self.position - neighbor.position) < 15:  # Increased radius for group-seeking behavior
                    group_center += neighbor.position
                    num_neighbors += 1

            if num_neighbors > 0:
                group_center /= num_neighbors
                direction_to_group = group_center - self.position
                distance_to_group = np.linalg.norm(direction_to_group)

                if distance_to_group > 0:
                    direction_to_group = direction_to_group / distance_to_group
                    self.velocity += direction_to_group * 1.0  # Increased strength of group attraction

        if self.goal is not None:
            direction_to_goal = self.goal - self.position
            distance_to_goal = np.linalg.norm(direction_to_goal)

            # Normalize direction to goal
            if distance_to_goal > 0:
                direction_to_goal = direction_to_goal / distance_to_goal
            else:
                direction_to_goal = np.array([0.0, 0.0])

            # Combine forces: First apply obstacle avoidance, then goal-seeking
            self.velocity += direction_to_goal * 0.5  # Apply goal force with a smaller weight to avoid overriding obstacle avoidance

            # Normalize velocity to match speed
            if np.linalg.norm(self.velocity) > 0:
                self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

    def avoid_obstacles(self, environment):
        """Detect obstacles in the path and apply a repulsion force to move around them."""
        # Raycast ahead based on the agent's velocity to detect obstacles in its path
        step_ahead = self.position + self.velocity * 1.0  # Predict the next step

        for obstacle in environment.obstacles:
            x1, y1, x2, y2 = obstacle
            # Check if the predicted path (step_ahead) intersects the obstacle
            if x1 <= step_ahead[0] <= x2 and y1 <= step_ahead[1] <= y2:
                # Collision detected: steer away from obstacle by adjusting the velocity
                self.steer_around_obstacle(obstacle)
                break  # Stop after applying the avoidance force

    def steer_around_obstacle(self, obstacle):
        """Apply a smooth steering force to avoid getting stuck."""
        # Calculate the center of the obstacle (for simplicity, use the middle of the obstacle)
        x1, y1, x2, y2 = obstacle
        obstacle_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Calculate the direction away from the obstacle center
        avoidance_direction = self.position - obstacle_center  # Direction away from the obstacle
        avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)  # Normalize

        # Apply the avoidance force to the velocity by rotating the direction
        angle_change = np.pi / 4  # 45 degrees rotation
        rotation_matrix = np.array([[np.cos(angle_change), -np.sin(angle_change)],
                                    [np.sin(angle_change), np.cos(angle_change)]])
        self.velocity = np.dot(rotation_matrix, self.velocity)

    def move(self):
        """Update the agent's position based on its velocity."""
        self.position += self.velocity

    def reached_goal(self):
        """Check if the agent has reached its goal."""
        if self.goal is not None:
            distance_to_goal = np.linalg.norm(self.position - self.goal)
            if distance_to_goal < 0.5:
                self.goal = None  # Keep the goal fixed, no removal of goal
        return self.goal is None  # Return True if the goal is reached

# Initialize the simulation
def initialize_simulation(num_agents, num_wanderers, num_group_seekers, num_path_followers, env_width, env_height):
    environment = Environment(
        width=env_width,
        height=env_height,
        obstacles=create_obstacles(env_width, env_height),  # Generate non-overlapping obstacles
        goals=[]  # Empty goal list (we'll fill it later)
    )
    
    goals = create_goals(environment)  # Create 5 fixed goals in the environment
    
    environment.goals = goals  # Assign the generated goals to the environment
    
    agents = []
    goal_index = 0
    for _ in range(num_agents):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        goal = goals[goal_index]  # Assign one of the fixed goals
        goal_index = (goal_index + 1) % len(goals)  # Cycle through the goals if there are more agents than goals
        agents.append(Agent(start_pos, goal, speed=0.5, personal_space=0.5))

    for _ in range(num_wanderers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        goal = goals[goal_index]  # Assign one of the fixed goals
        goal_index = (goal_index + 1) % len(goals)
        agents.append(Agent(start_pos, goal, wanderer=True))

    for _ in range(num_group_seekers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        goal = goals[goal_index]  # Assign one of the fixed goals
        goal_index = (goal_index + 1) % len(goals)
        agents.append(Agent(start_pos, goal, group_seeker=True))

    for _ in range(num_path_followers):
        start_pos = (random.uniform(0, env_width), random.uniform(0, env_height))
        goal = goals[goal_index]  # Assign one of the fixed goals
        goal_index = (goal_index + 1) % len(goals)
        agents.append(Agent(start_pos, goal, path_follower=True))

    return environment, agents

# Function to create fixed goals
def create_goals(environment):
    goals = []
    # Define 5 fixed goals in the environment, checking if they are not inside obstacles
    for _ in range(5):
        valid_goal = False
        while not valid_goal:
            goal = np.array([random.uniform(1, 19), random.uniform(1, 19)])  # Random goal position
            valid_goal = environment.is_goal_valid(goal)  # Check if the goal is outside obstacles
        goals.append(goal)
    return goals

# Function to create more obstacles (buildings)
def create_obstacles(width, height, num_obstacles=15):
    obstacles = []
    
    while len(obstacles) < num_obstacles:
        # Generate a new obstacle with random size and position
        x1 = random.uniform(0, width - 2)
        y1 = random.uniform(0, height - 2)
        x2 = x1 + random.uniform(0.5, 2)  # Smaller width of the building
        y2 = y1 + random.uniform(0.5, 2)  # Smaller height of the building
        
        # Check if the new obstacle overlaps with any existing obstacle
        overlap = False
        for obstacle in obstacles:
            if not (x2 <= obstacle[0] or x1 >= obstacle[2] or y2 <= obstacle[1] or y1 >= obstacle[3]):
                # If the new obstacle intersects with an existing one, set overlap to True
                overlap = True
                break
        
        if not overlap:
            # If no overlap, add the new obstacle to the list
            obstacles.append((x1, y1, x2, y2))

    return obstacles

# Add new agents at spawn points
def add_new_agents(spawn_points, agents, num_new_agents, env_width, env_height):
    for _ in range(num_new_agents):
        spawn_point = random.choice(spawn_points)
        goal = np.array([random.uniform(0, 20), random.uniform(0, 20)])  # Only assign one goal
        agents.append(Agent(spawn_point, goal, speed=0.5, personal_space=1.5))

# Run the simulation
import os

# Run the simulation and save images for each step
def run_simulation(environment, agents, spawn_points, steps=100, save_dir="simulation_images"):
    # Create the directory to save images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 8))  # Increased figure size to make space for the legend

    # Legend handles for obstacles, agents, wanderers, etc.
    obstacles_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Obstacle')
    agents_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Agent')
    wanderers_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Wanderer')  # Smaller size for wanderers
    goals_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Goal')
    spawn_patch = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='Spawn Point')

    for step in range(steps):
        plt.clf()  # Clear the figure but the legend will stay intact

        # Draw environment
        plt.xlim(0, environment.width)
        plt.ylim(0, environment.height)
        for obstacle in environment.obstacles:
            # Draw only the outlines of the obstacles (rectangles)
            x_vals = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
            y_vals = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
            plt.plot(x_vals, y_vals, color='black', lw=2)  # Outline of the obstacle

        # Draw goals (fixed goals and other goals with different colors)
        for goal in environment.goals:
            plt.scatter(goal[0], goal[1], color='red', s=100, marker='o')  # Fixed goals are in red

        # Draw spawn points with a distinct color and marker
        for spawn in spawn_points:
            plt.scatter(spawn[0], spawn[1], color='purple', s=100, label="Spawn Point", marker='D')  # Diamond marker for spawn points

        # Add new agents at spawn points every 10 steps
        if step % 10 == 0:
            add_new_agents(spawn_points, agents, num_new_agents=3, env_width=environment.width, env_height=environment.height)

        # Update agents
        for agent in agents:
            neighbors = [a for a in agents if a != agent and np.linalg.norm(a.position - agent.position) < agent.personal_space]
            agent.compute_velocity(neighbors, environment)
            agent.move()

            # Draw agent (distinguishing wanderers with smaller size)
            if agent.wanderer:
                plt.scatter(agent.position[0], agent.position[1], color='blue', s=20)  # Wanderers are blue and smaller
            else:
                plt.scatter(agent.position[0], agent.position[1], color='blue', s=50)  # Normal agents are blue and larger
            if agent.goal is not None:
                plt.scatter(agent.goal[0], agent.goal[1], color='red', s=50, alpha=0.5)  # Draw current goal

            if agent.reached_goal():
                agent.goal = None  # Keep the goal fixed, no removal of goal

        # Create legend after agents are drawn and the figure is updated
        plt.legend(handles=[obstacles_patch, agents_patch, wanderers_patch, goals_patch, spawn_patch], loc='upper left', bbox_to_anchor=(1.05, 1))

        # Adjust the layout to make space for the legend
        plt.subplots_adjust(right=0.8)  # Reduce the right margin to make space for the legend

        # Save the current plot as an image
        image_filename = os.path.join(save_dir, f"step_{step:03d}.png")  # Save with a step number (e.g., step_000.png)
        plt.savefig(image_filename, bbox_inches='tight')  # Save the image with tight layout

        plt.pause(0.1)

    plt.show()

# Main execution
if __name__ == "__main__":
    env_width, env_height = 20, 20
    spawn_points = [(2, 2), (18, 18), (15, 3)]  # Locations where new agents can spawn
    env, agents = initialize_simulation(num_agents=50, num_wanderers=20, num_group_seekers=25, num_path_followers=25, env_width=env_width, env_height=env_height)
    run_simulation(env, agents, spawn_points)
