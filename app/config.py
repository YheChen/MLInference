# app/config.py

MODEL_PATH = "app/model/model.pkl"

# Number of features the trained model expects. Requests are validated against
# this at the API boundary so malformed input can never reach the batcher.
FEATURE_DIM = 10

BATCH_MAX_SIZE = 32
BATCH_WINDOW_MS = 5

QUEUE_MAX_SIZE = 2000
QUEUE_HIGH_WATERMARK = int(QUEUE_MAX_SIZE * 0.8)

REQUEST_TIMEOUT_MS = 100  # hard timeout
