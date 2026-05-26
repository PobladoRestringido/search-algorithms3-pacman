import subprocess
from utils import parse_pacman_output
from collections import defaultdict


def run_pacman(agent: str, layout: str | None = None):
    """
    Runs a single Pacman game with the given agent and layout.
    Returns the raw output text.

    If layout is None, a default layout is used.
    """

    layout = layout or "mediumClassic"  # default layout

    cmd = ["python", "pacman.py", "-p", agent, "-l", layout, "-q"]  # quiet graphics

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    return result.stdout


def sweep(
    agents: list[str],
    layouts: list[str | None],
    runs: int,
):
    """
    Runs all combinations of agents × layouts for a given number of runs.

    Parameters
    ----------
    agents : list[str]
        Agent class names (e.g., ["NeuralAgent", "ReflexAgent"])
    layouts : list[str]
        Layout names (e.g., ["smallClassic", "mediumClassic"])
    runs : int
        Number of runs per (agent, layout) pair.

    Returns
    -------
    results : dict
        results[(agent, layout)] = list of parsed run dictionaries
    """
    results = defaultdict(list)

    for agent in agents:
        for layout in layouts:
            print(f"\n=== {agent} on {layout} ({runs} runs) ===")

            for i in range(runs):
                print(f"  Run {i+1}/{runs}...", end=" ")

                actual_layout = layout or "mediumClassic"  # default layout

                raw = run_pacman(agent, layout)
                parsed = parse_pacman_output(raw)

                results[(agent, layout)].append(parsed)

                print("done")

    return results


def print_summary_table(results: dict) -> None:
    """
    Prints a neat summary table from the sweep results.
    """
    print("\n==================== SUMMARY ====================")
    print(f"{'Agent':15} {'Layout':15} {'Runs':5} {'AvgScore':10} {'WinRate':8}")

    for (agent, layout), runs in results.items():
        scores = [r.get("score", 0) for r in runs]
        wins = sum(1 for r in runs if r.get("record") == "Win")
        total = len(runs)

        avg_score = sum(scores) / total if total else 0
        win_rate = wins / total if total else 0

        print(f"{agent:15} {layout:15} {total:<5} {avg_score:<10.1f} {win_rate:<8.2f}")


def benchmark(
    agents: list[str],
    layouts: list[str | None],
    n_runs: int,
) -> dict:
    """
    Full end‑to‑end orchestrator:
    - runs all sweeps
    - collects results
    - prints summary table
    - returns results for further processing
    """
    results = sweep(agents, layouts, n_runs)
    print_summary_table(results)
    return results
