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

        # Score
        self.assertIn("scores", result)
        self.assertEqual(result["scores"], [-471])

    def test_parse_game_index(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("game_indices", result)
        self.assertEqual(result["game_indices"], [22])

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
        self.assertEqual(result["win_rate"], "0/1 (0.00)")

    def test_parse_record(self):
        result = parse_pacman_output(self.sample_output)

        self.assertIn("record", result)
        self.assertEqual(result["record"], ["Loss"])

    def test_parses_batched_output(self):

        batched_output = """Modelo cargado correctamente desde models/only_human_games.pth
Tamaño de entrada: (20, 11)
NeuralAgent inicializado, usando dispositivo: cuda
Reply mode: False
Pacman died! Score: -420
Datos del juego 185 guardados en pacman_data/game_185.csv
Pacman died! Score: -234
Datos del juego 186 guardados en pacman_data/game_186.csv
Pacman died! Score: -349
Datos del juego 187 guardados en pacman_data/game_187.csv
Pacman died! Score: 283
Datos del juego 188 guardados en pacman_data/game_188.csv
Pacman died! Score: 428
Datos del juego 189 guardados en pacman_data/game_189.csv
Average Score: -58.4
Scores:        -420.0, -234.0, -349.0, 283.0, 428.0
Win Rate:      0/5 (0.00)
Record:        Loss, Loss, Loss, Loss, Loss"""

        parsed: dict = parse_pacman_output(batched_output)

        # Expected values
        expected_indices = [185, 186, 187, 188, 189]
        expected_all_scores = [-420.0, -234.0, -349.0, 283.0, 428.0]
        expected_avg_score = -58.4
        expected_n_wins = 0
        expected_n_games = 5
        expected_winrate = "0/5 (0.00)"
        expected_record = ["Loss", "Loss", "Loss", "Loss", "Loss"]

        # Assertions
        self.assertEqual(parsed["game_indices"], expected_indices)
        self.assertEqual(parsed["scores"], expected_all_scores)
        self.assertAlmostEqual(parsed["average_score"], expected_avg_score)
        self.assertEqual(parsed["wins"], expected_n_wins)
        self.assertEqual(parsed["games"], expected_n_games)
        self.assertEqual(parsed["win_rate"], expected_winrate)
        self.assertEqual(parsed["record"], expected_record)


if __name__ == "__main__":
    unittest.main()
