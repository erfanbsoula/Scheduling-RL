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
MODEL_PATH = 'saves/'

BUFFER_SIZE = 1e6
EXPLORATION_EPISODES = 30
MAX_EPISODES = 2000
MAX_STEPS = 10000
UPDATE_INTERVAL = 100
BATCH_SIZE = 64
Q_LEARNING_RATE = 1e-2
POLICY_LEARNING_RATE = 1e-3
DISCOUNT_RATE = 0.95
SOFT_UPDATE_TAU = 1e-3
TARGET_UPDATE_DELAY = 1
UPDATE_REPEAT_COUNT = 1
CHECKPOINT_INTERVAL = 100

########################################
# Simulation parameters for env.py

PROCESSOR_COUNT = 8
TASK_PER_PROCESSOR = 5
INSTANCES_PER_TASK = 20

MIN_TASK_AVG_LOG_EXEC_TIME = 2.8
MAX_TASK_AVG_LOG_EXEC_TIME = 3.5

MIN_TASK_SIGMA_LOG_EXEC_TIME = 0.3
MAX_TASK_SIGMA_LOG_EXEC_TIME = 0.5

MIN_DEADLINE_FACTOR = 1.2
MAX_DEADLINE_FACTOR = 2.0

MIN_LOAD = 0.5
MAX_LOAD = 1.2

STATIC_POWER_COEFF = 0.3
DYNAMIC_POWER_COEFF = 0.7
ENERGY_PENALTY_COEFF = 0.05

########################################