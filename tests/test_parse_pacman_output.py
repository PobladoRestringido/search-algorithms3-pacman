import unittest
from utils import parse_pacman_output


class TestPacmanOutputParser(unittest.TestCase):

    def setUp(self):

        self.sample_output = """
Modelo cargado correctamente desde models/pacman_model.pth
Tamaño de entrada: (20, 11)
NeuralAgent inicializado, usando dispositivo: cuda
Reply mode: False
Pacman died! Score: -471
Datos del juego 22 guardados en pacman_data/game_22.csv
Average Score: -471.0
Scores:        -471.0
Win Rate:      0/1 (0.00)
Record:        Loss
"""

    def test_parse_basic_fields(self):
        result = parse_pacman_output(self.sample_output)

        # Basic correctness
        self.assertIsInstance(result, dict)

        # Death detection
        self.assertIn("died", result)
        self.assertTrue(result["died"])

        # Score
        self.assertIn("score", result)
        self.assertEqual(result["score"], -471)

    def test_parse_game_index(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("game_index", result)
        self.assertEqual(result["game_index"], 22)

    def test_parse_average_score(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("average_score", result)
        self.assertAlmostEqual(result["average_score"], -471.0)

    def test_parse_scores_list(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("scores", result)
        self.assertEqual(result["scores"], [-471.0])

    def test_parse_win_rate(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("wins", result)
        self.assertIn("games", result)
        self.assertIn("win_rate", result)

        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["games"], 1)
        self.assertAlmostEqual(result["win_rate"], 0.00)

    def test_parse_record(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("record", result)
        self.assertEqual(result["record"], "Loss")


if __name__ == "__main__":
    unittest.main()
