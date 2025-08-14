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
SAVE_PATH = 'saves/try3/'

BUFFER_SIZE = 1e6
MAX_EPISODES = 1000
MAX_STEPS = 20000
UPDATE_INTERVAL = 100
UPDATE_REPEAT_COUNT = 1
BATCH_SIZE = 32
Q_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 1e-3
TARGET_UPDATE_DELAY = 1
SOFT_UPDATE_TAU = 1e-3
DISCOUNT_RATE = 0.999
CHECKPOINT_INTERVAL = 100

########################################
# Simulation parameters for env.py

PROCESSOR_COUNT = 4
TASK_PER_PROCESSOR = 5
INSTANCES_PER_TASK = 8

MIN_LOAD = 0.4
MAX_LOAD = 0.9

MIN_PERIOD = 100
MAX_PERIOD = 1000

STATIC_POWER_COEFF = 0.3
DYNAMIC_POWER_COEFF = 0.7
ENERGY_PENALTY_COEFF = 0.01

########################################