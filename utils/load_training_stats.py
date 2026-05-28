import os
import pandas as pd


def load_training_stats(folder):
    """
    Reads all CSV game files in a folder and returns:
    - num_games: number of CSV files
    - avg_score: average final score across games
    """
    scores = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            # final score = score of last row
            scores.append(df["score"].iloc[-1])

    if not scores:
        return {"num_games": 0, "avg_score": 0.0}

    return {
        "num_games": len(scores),
        "avg_score": sum(scores) / len(scores),
    }
