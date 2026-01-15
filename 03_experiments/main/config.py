MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10             ### 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RANDOM_SEED = 42

DATA_PATH = "./dev_phase/subtask1/train"
OUTPUT_DIR = "./results/xlmr5"
CSV_LOG_FILE = "./results/logs/xlmr5_logs.csv"