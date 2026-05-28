import os
import shutil
import tempfile
import unittest
import pandas as pd

from utils import load_training_stats


class TestTrainingStats(unittest.TestCase):
    """Tests for the load_training_stats() helper function."""

    def setUp(self):
        """Create a temporary directory for mock CSV game files."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.tmpdir)

    def _write_csv(self, filename, scores):
        """
        Helper to write a mock game CSV.
        'scores' is a list of score values for each row.
        """
        df = pd.DataFrame(
            {
                "timestamp": ["t"] * len(scores),
                "agent_index": [0] * len(scores),
                "action": ["Stop"] * len(scores),
                "score": scores,
                "is_win": [False] * len(scores),
                "is_lose": [False] * len(scores),
                "game_over": [False] * len(scores),
                "map_matrix": ["[]"] * len(scores),
            }
        )
        df.to_csv(os.path.join(self.tmpdir, filename), index=False)

    def test_empty_folder(self):
        """If the folder has no CSV files, return zero games and zero average score."""
        stats = load_training_stats(self.tmpdir)
        self.assertEqual(stats["num_games"], 0)
        self.assertEqual(stats["avg_score"], 0.0)

    def test_single_game(self):
        """Correctly compute stats for a single CSV game."""
        self._write_csv("game1.csv", [0, 10, 20])
        stats = load_training_stats(self.tmpdir)
        self.assertEqual(stats["num_games"], 1)
        self.assertEqual(stats["avg_score"], 20.0)  # last row score

    def test_multiple_games(self):
        """Correctly compute average score across multiple CSV games."""
        self._write_csv("game1.csv", [0, 10, 20])  # final = 20
        self._write_csv("game2.csv", [5, 15, 25])  # final = 25
        self._write_csv("game3.csv", [7, 8, 9])  # final = 9

        stats = load_training_stats(self.tmpdir)

        self.assertEqual(stats["num_games"], 3)
        self.assertAlmostEqual(stats["avg_score"], (20 + 25 + 9) / 3)

    def test_ignores_non_csv_files(self):
        """Non-CSV files must be ignored."""
        self._write_csv("game1.csv", [0, 10])
        with open(os.path.join(self.tmpdir, "notes.txt"), "w") as f:
            f.write("not a game")

        stats = load_training_stats(self.tmpdir)
        self.assertEqual(stats["num_games"], 1)
        self.assertEqual(stats["avg_score"], 10.0)


if __name__ == "__main__":
    unittest.main()
