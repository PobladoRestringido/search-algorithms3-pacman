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

        - 'scores' : list[float]
            The game scores extracted from lines like "Score: -471".

        - 'game_indices' : list[int]
            The game number extracted from lines such as
            "Datos del juego 22 guardados ...".

        - 'average_score' : float
            The average score across runs, from lines like
            "Average Score: -471.0".

        - 'wins' : int
            Number of wins extracted from the "Win Rate:" summary.

        - 'games' : int
            Total number of games played, also from the "Win Rate:" summary.

        - 'win_rate' : str
            Win percentage as a string, e.g. '0/5 (0.00)'.

        - 'record' : list[str]
            The textual record, typically "Win" or "Loss", extracted from
            the "Record:" line.

    """

    data = {}

    # 1. Game indices
    game_idx_matches = re.findall(r"Datos del juego\s+(\d+)", text)
    data["game_indices"] = [int(x) for x in game_idx_matches]

    # 2. Individual scores
    score_matches = re.findall(r"(?:Pacman (?:died|won)!)\s*Score:\s*(-?\d+)", text)
    data["scores"] = [float(s) for s in score_matches]

    # 3. Average score
    m = re.search(r"Average Score:\s*(-?\d+\.?\d*)", text)
    if m:
        data["average_score"] = float(m.group(1))

    # 4. Win Rate
    m = re.search(r"Win Rate:\s*(\d+/\d+\s*\([\d\.]+\))", text)
    if m:
        winrate_literal = m.group(1)
        data["win_rate"] = winrate_literal

        # Extract wins and games separately
        wins, games = re.match(r"(\d+)/(\d+)", winrate_literal).groups()
        data["wins"] = int(wins)
        data["games"] = int(games)

    # 5. Record list
    m = re.search(r"Record:\s*(.*)", text)
    if m:
        record_str = m.group(1)
        records = [r.strip() for r in record_str.split(",")]
        data["record"] = records

    return data
