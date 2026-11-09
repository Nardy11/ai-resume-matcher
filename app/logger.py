import os

LOG_FILE = "logs/app.log"


def log_inference_time(time_taken):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"Inference time: {time_taken}\n")
