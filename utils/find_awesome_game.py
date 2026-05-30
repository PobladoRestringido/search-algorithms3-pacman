from utils import parse_pacman_output, run_pacman
import os
import random


def find_awesome_game(
    agent: str = "AlphaBetaNeuralAgent",
    layout: str = "mediumClassic",
    model_path: str = "models/only_human_games.pth",
    n_runs: int = 100,
) -> dict:
    """
    Generates a bunch of games from scratch and keeps track of the id and seed of the
    best one so far.
    """
    best_score = float("-inf")
    best_seed = None
    best_index = None

    for i in range(n_runs):

        # Generate a fresh random seed for this run
        seed = random.randint(0, 10_000_000)

        # Inject it into the environment so Pacman uses it
        os.environ["PACMAN_SEED"] = str(seed)

        output_raw = run_pacman(
            agent=agent, layout=layout, model_path=model_path, fast=True, n_runs=1
        )
        parsed_output: dict = parse_pacman_output(output_raw)

        score: float = parsed_output["average_score"]

        print(f"[{i}] seed={seed} → score={score}")

        if score is not None and score > best_score:
            best_score = score
            best_seed = seed
            best_index = i

    return {
        "best_score": best_score,
        "best_seed": best_seed,
        "best_index": best_index,
    }
