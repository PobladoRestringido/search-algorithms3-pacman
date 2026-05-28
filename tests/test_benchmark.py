import unittest
from utils import benchmark


class TestBenchmark(unittest.TestCase):

    def test_benchmark_structure(self):
        agents = ["ReflexAgent"]
        layouts = ["smallClassic"]
        neural_net_paths = [None]
        n_runs = 2

        results = benchmark(
            agents, layouts, n_runs=n_runs, fast=True, neural_net_paths=neural_net_paths
        )

        self.assertIsInstance(results, dict)
        # self.assertEqual(len(results), 1)

        key = ("ReflexAgent", "smallClassic", "default model")
        self.assertIn(key, results)

        parsed_output = results[key]
        # self.assertEqual(len(runs), n_runs)

        self.assertIsInstance(parsed_output, dict)

    def test_benchmark_expected_fields(self):
        agents = ["ReflexAgent"]
        layouts = ["mediumClassic"]
        neural_net_paths = [None]
        n_runs = 1

        results = benchmark(
            agents, layouts, n_runs=n_runs, fast=True, neural_net_paths=neural_net_paths
        )
        # breakpoint()
        parsed_output = results[("ReflexAgent", "mediumClassic", "default model")]

        self.assertIn("scores", parsed_output)
        self.assertIn("record", parsed_output)

        scores = parsed_output["scores"]
        self.assertIsInstance(scores, list)
        for score in scores:
            self.assertIsInstance(score, (int, float))

        record = parsed_output["record"]
        self.assertIsInstance(record, list)
        for r in record:
            self.assertIsInstance(r, str)

    def test_benchmark_multiple_inputs_expected_fields(self):
        agents = ["NeuralAgent", "NeuralAgent"]
        layouts = [None, None]  # default layout
        neural_net_paths = [None, "models/only_human_games.pth"]
        n_runs = 2

        benchmark_results: dict[tuple[str, str, str], dict] = benchmark(
            agents=agents,
            layouts=layouts,
            neural_net_paths=neural_net_paths,
            n_runs=n_runs,
        )

        for key, parsed_output in benchmark_results.items():
            self.assertIsInstance(key, tuple)
            for e in key:
                self.assertIsInstance(e, str)

            self.assertIsInstance(parsed_output, dict)

            self.assertIn("game_indices", parsed_output)
            game_indices = parsed_output["game_indices"]
            self.assertNotEqual(len(game_indices), 0)

            self.assertIn("scores", parsed_output)
            scores = parsed_output["scores"]
            self.assertNotEqual(len(scores), 0)

            self.assertIn("average_score", parsed_output)
            self.assertIn("win_rate", parsed_output)
            self.assertIn("wins", parsed_output)
            self.assertIn("games", parsed_output)
            self.assertIn("record", parsed_output)


if __name__ == "__main__":
    unittest.main()
