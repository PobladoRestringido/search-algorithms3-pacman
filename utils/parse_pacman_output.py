import re


def parse_pacman_output(text: str) -> dict:
    """
        Extracts relevant metrics from the Pacman engine output using regex.
        Returns a dictionary with all detected fields.

        Parameters
    ----------
    text : str
        The full textual output produced during a Pacman game run.

    Returns
    -------
    results : dict
        A dictionary containing the parsed metrics. Keys include:

        - 'died' : bool
            True if the output contains the phrase "Pacman died!", otherwise False.

        - 'score' : int
            The final game score extracted from lines like "Score: -471".

        - 'game_index' : int
            The game number extracted from lines such as
            "Datos del juego 22 guardados ...".

        - 'average_score' : float
            The average score across runs, from lines like
            "Average Score: -471.0".

        - 'scores' : list of float
            A list of individual scores extracted from the "Scores:" line.

        - 'wins' : int
            Number of wins extracted from the "Win Rate:" summary.

        - 'games' : int
            Total number of games played, also from the "Win Rate:" summary.

        - 'win_rate' : float
            Win percentage as a decimal, e.g. 0.00 for 0%.

        - 'record' : str
            The textual record, typically "Win" or "Loss", extracted from
            the "Record:" line.

    """

    data = {}

    # Pacman died
    data["died"] = bool(re.search(r"Pacman died", text))

    # Final score
    m = re.search(r"Score:\s*(-?\d+)", text)
    if m:
        data["score"] = int(m.group(1))

    # Game index
    m = re.search(r"Datos del juego\s+(\d+)", text)
    if m:
        data["game_index"] = int(m.group(1))

    # Average Score
    m = re.search(r"Average Score:\s*(-?\d+\.?\d*)", text)
    if m:
        data["average_score"] = float(m.group(1))

    # Scores list
    m = re.search(r"Scores:\s*(-?\d+\.?\d*)", text)
    if m:
        data["scores"] = [float(m.group(1))]

    # Win Rate
    m = re.search(r"Win Rate:\s*(\d+)/(\d+)\s*\(([\d\.]+)\)", text)
    if m:
        data["wins"] = int(m.group(1))
        data["games"] = int(m.group(2))
        data["win_rate"] = float(m.group(3))

    # Record: Win / Loss
    m = re.search(r"Record:\s*(\w+)", text)
    if m:
        data["record"] = m.group(1)

    return data
