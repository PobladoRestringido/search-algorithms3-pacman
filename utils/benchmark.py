import subprocess
from utils import parse_pacman_output
from collections import defaultdict


class ModelNotFoundError(Exception):
    pass


def run_pacman(
    agent: str,
    layout: str | None = None,
    fast: bool = False,
    model_path: str | None = None,
    n_runs: int = 1,
):
    """
    Runs a single Pacman game with the given agent and layout.
    Returns the raw output text.

    Parameters
    ----------
    agent : str
        Name of the Pacman agent class.
    layout : str | None
        Layout name (default: mediumClassic)
    fast : bool
        If True, run with --frameTime 0 for maximum speed.

    Raises
    ------
    ModelNotFoundError
        If `model_path` cannot be found.
    """

    cmd = ["python", "pacman.py", "-p", agent, "-q", "-n", str(n_runs)]

    if layout:
        cmd += ["-l", layout]

    if model_path:
        cmd += ["-a", f"model_path={model_path}"]

    if fast:
        cmd += ["--frameTime", "0"]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    if "ERROR: No se encontró el modelo" in result.stdout:
        raise ModelNotFoundError(f"Model with path {model_path} not found")

    return result.stdout


def sweep(
    agents: list[str],
    layouts: list[str | None],
    neural_net_paths: list[str | None],
    runs: int,
    fast: bool = False,
) -> dict[tuple[str, str, str], dict]:
    """
    Runs all combinations of agents × layouts for a given number of runs.

    Parameters
    ----------
    agents : list[str]
    layouts : list[str]
    runs : int
    fast : bool
        If True, pass fast=True to run_pacman for ultra‑quick runs.
    """
    assert len(agents) == len(layouts) == len(neural_net_paths)

    results = {}

    for agent, layout, model_path in zip(agents, layouts, neural_net_paths):
        #breakpoint()
        pretty_layout = layout or "default layout"
        pretty_model = model_path or "default model"

        print(
            f"\n=== Running {agent} on {pretty_layout} using {pretty_model} ({runs} runs) ==="
        )

        raw = run_pacman(agent, layout, fast=fast, model_path=model_path, n_runs=runs)
        parsed: dict = parse_pacman_output(raw)

        key = (agent, pretty_layout, pretty_model)
        results[key] = parsed

        print("done")

    return results


def print_summary_table(results: dict[tuple[str, str, str], dict]) -> None:
    """
    Prints a neat summary table from the sweep results.

    Summary table has the following columns:
    - Agent
    - Layout
    - Model Path
    - Runs
    - Avg Score
    - WinRate
    """
    MODEL_PATH_WIDTH = 35

    print(f"\n{'='*40} BENCHMARK {'='*40}")
    print(
        f"{'Agent':15} {'Layout':15} {'Model Path':{MODEL_PATH_WIDTH}} {'Runs':5} {'AvgScore':10} {'WinRate':8}"
    )

    for (agent, layout, model_path), parsed_dict in results.items():

        total_runs = parsed_dict.get("games", 0)
        avg_score = parsed_dict.get("average_score", 0.0)
        win_rate = parsed_dict.get("win_rate", 0.0)

        print(
            f"{agent:15} {layout:15} {model_path:{MODEL_PATH_WIDTH}} {total_runs:<5} {avg_score:<10.1f} {win_rate:8}"
        )


def benchmark(
    agents: list[str],
    layouts: list[str | None],
    neural_net_paths: list[str | None],
    n_runs: int,
    fast: bool = False,
) -> dict:
    """
    Full end‑to‑end orchestrator:
    - runs all sweeps
    - collects results
    - prints summary table
    - returns results for further processing

    Parameters
    ----------
    agents: list[str]
        List of agent names to be used.

    layouts: list[str | None]
        List of layouts to be employed. Use `None`'s for default layout.

    neural_net_paths: list[str | None]
        List of paths to the trained neural net models to be used. Use `None`'s for
        "models/pacman_model.pth".

    n_runs: int
        Number of runs to be performed with each agent.

    fast: bool
        (I think this parameter doesn't work) - Pablo
    """
    results = sweep(
        agents, layouts, runs=n_runs, fast=fast, neural_net_paths=neural_net_paths
    )
    print_summary_table(results)
    return results
