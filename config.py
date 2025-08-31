import os

########################################
# Device configuration parameters

GPU = False
DEVICE_INDEX = 0

########################################
# General parameters for train.py

DVFS_LEVELS = [0.25, 0.5, 0.75, 1.0]

CRITIC_STATE_DIM = 16
ACTOR_STATE_DIM = 6
ACTION_DIM = 2
HIDDEN_DIM = [16, 8]
SAVE_PATH = 'saves/try4/'

BUFFER_SIZE = 1e5
MAX_EPISODES = 1000
MAX_STEPS = 10000
UPDATE_INTERVAL = 32
UPDATE_REPEAT_COUNT = 1
BATCH_SIZE = 32
Q_LEARNING_RATE = 1e-4
POLICY_LEARNING_RATE = 1e-4
TARGET_UPDATE_DELAY = 1
SOFT_UPDATE_TAU = 1e-3
DISCOUNT_RATE = 0.999
CHECKPOINT_INTERVAL = 100

########################################
# Simulation parameters for env.py

PROCESSOR_COUNT = 4
TASK_PER_PROCESSOR = 5
MAX_EPISODE_TIME = 1000

MIN_LOAD = 0.65
MAX_LOAD = 0.75

MIN_PERIOD = 10
MAX_PERIOD = 100

INSTANCE_COMPLETION_REWARD = 0.1
INSTANCE_MISS_PENALTY = 2.0

STATIC_POWER_COEFF = 0.3
DYNAMIC_POWER_COEFF = 0.7
ENERGY_PENALTY_COEFF = 0.05

########################################
# For grid search launches from cli

SAVE_PATH = os.getenv('GRID_SAVE_PATH', SAVE_PATH)
Q_LEARNING_RATE = float(os.getenv('GRID_Q_LEARNING_RATE', 1e-4))
POLICY_LEARNING_RATE = float(os.getenv('GRID_POLICY_LEARNING_RATE', 1e-4))
SOFT_UPDATE_TAU = float(os.getenv('GRID_SOFT_UPDATE_TAU', 1e-3))
DISCOUNT_RATE = float(os.getenv('GRID_DISCOUNT_RATE', 0.999))
INSTANCE_COMPLETION_REWARD = float(os.getenv('GRID_INSTANCE_COMPLETION_REWARD', 0.1))
INSTANCE_MISS_PENALTY = float(os.getenv('GRID_INSTANCE_MISS_PENALTY', 2.0))
ENERGY_PENALTY_COEFF = float(os.getenv('GRID_ENERGY_PENALTY_COEFF', 0.005))
