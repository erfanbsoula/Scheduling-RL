########################################
# Device configuration parameters

GPU = False
DEVICE_INDEX = 0

########################################
# General parameters for train.py

DVFS_LEVELS = [0.25, 0.5, 0.75, 1.0]

STATE_DIM = 10
ACTION_DIM = 2
HIDDEN_DIM = [8, 4]
SAVE_PATH = 'saves/try1/'

BUFFER_SIZE = 1e6
EXPLORATION_EPISODES = 20
EXPLORATION_PROBABILITY = 0.05
MAX_EPISODES = 1000
MAX_STEPS = 10000
UPDATE_INTERVAL = 100
BATCH_SIZE = 64
Q_LEARNING_RATE = 1e-2
POLICY_LEARNING_RATE = 1e-3
DISCOUNT_RATE = 0.99
SOFT_UPDATE_TAU = 1e-3
TARGET_UPDATE_DELAY = 3
UPDATE_REPEAT_COUNT = 3
CHECKPOINT_INTERVAL = 50

########################################
# Simulation parameters for env.py

PROCESSOR_COUNT = 1
TASK_PER_PROCESSOR = 4
INSTANCES_PER_TASK = 5

MIN_LOAD = 0.4
MAX_LOAD = 1.5

MIN_PERIOD = 100
MAX_PERIOD = 1000

STATIC_POWER_COEFF = 0.3
DYNAMIC_POWER_COEFF = 0.7
ENERGY_PENALTY_COEFF = 0.01

########################################