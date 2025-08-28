########################################
# Device configuration parameters

GPU = False
DEVICE_INDEX = 0

########################################
# General parameters for train.py

DVFS_LEVELS = [0.25, 0.5, 0.75, 1.0]

STATE_DIM = 16
ACTION_DIM = 2
HIDDEN_DIM = [16, 8]
SAVE_PATH = 'saves/try4/'

BUFFER_SIZE = 1e5
MAX_EPISODES = 1000
MAX_STEPS = 1000
UPDATE_INTERVAL = 32
UPDATE_REPEAT_COUNT = 1
BATCH_SIZE = 32
Q_LEARNING_RATE = 1e-4
POLICY_LEARNING_RATE = 1e-4
TARGET_UPDATE_DELAY = 1
SOFT_UPDATE_TAU = 1e-3
DISCOUNT_RATE = 0.99
CHECKPOINT_INTERVAL = 100

########################################
# Simulation parameters for env.py

PROCESSOR_COUNT = 4
TASK_PER_PROCESSOR = 5
INSTANCES_PER_TASK = 16

MIN_LOAD = 1.1
MAX_LOAD = 1.3

MIN_PERIOD = 100
MAX_PERIOD = 1000

STATIC_POWER_COEFF = 0.2
DYNAMIC_POWER_COEFF = 0.8
ENERGY_PENALTY_COEFF = 0.0

########################################