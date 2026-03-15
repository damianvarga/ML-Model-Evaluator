import pandas as pd
from datetime import datetime

def log_experiment(model_name, accuracy):

    data = {
        "model": model_name,
        "accuracy": accuracy,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([data])

    try:
        old = pd.read_csv("experiments.csv")
        df = pd.concat([old, df])
    except:
        pass

    df.to_csv("experiments.csv", index=False)
