import unittest
from utils import benchmark


class TestBenchmark(unittest.TestCase):

    def test_benchmark_structure(self):
        agents = ["ReflexAgent"]
        layouts = ["smallClassic"]
        n_runs = 2

        results = benchmark(agents, layouts, n_runs)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)

        key = ("ReflexAgent", "smallClassic")
        self.assertIn(key, results)

        runs = results[key]
        self.assertEqual(len(runs), n_runs)

        for r in runs:
            self.assertIsInstance(r, dict)

    def test_benchmark_expected_fields(self):
        agents = ["ReflexAgent"]
        layouts = ["mediumClassic"]
        n_runs = 1

        results = benchmark(agents, layouts, n_runs)
        run = results[("ReflexAgent", "mediumClassic")][0]

        self.assertIn("score", run)
        self.assertIn("record", run)

        self.assertIsInstance(run["score"], (int, float))

        self.assertIsInstance(run["record"], str)


if __name__ == "__main__":
    unittest.main()
