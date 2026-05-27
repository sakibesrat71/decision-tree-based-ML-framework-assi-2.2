import unittest

import numpy as np

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


class TestStreamingPreprocessing(unittest.TestCase):
    def test_standard_scaler_partial_fit_matches_batch_mean(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()

        scaler.partial_fit(X[:1])
        scaler.partial_fit(X[1:])

        np.testing.assert_allclose(scaler.mean, np.mean(X, axis=0))
        np.testing.assert_allclose(scaler.var, np.var(X, axis=0))

    def test_standard_scaler_transform_centres_data(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        transformed = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(np.mean(transformed, axis=0), [0, 0], atol=1e-8)

    def test_minmax_scaler_partial_fit_tracks_extremes(self):
        X = np.array([[2, 5], [0, 10], [4, 7]], dtype=float)
        scaler = MinMaxScaler()

        scaler.partial_fit(X[:1])
        scaler.partial_fit(X[1:])

        np.testing.assert_array_equal(scaler.min, [0, 5])
        np.testing.assert_array_equal(scaler.max, [4, 10])

    def test_simple_imputer_constant_fill(self):
        imputer = SimpleImputer(fill_value=-1)
        result = imputer.fit_transform([[1.0, np.nan]])

        np.testing.assert_array_equal(result, [[1.0, -1.0]])

    def test_simple_imputer_running_mean(self):
        imputer = SimpleImputer(strategy="mean")

        imputer.partial_fit([[1.0], [np.nan]])
        imputer.partial_fit([[3.0]])

        np.testing.assert_allclose(imputer.transform([[np.nan]]), [[2.0]])

    def test_one_hot_encoder_expands_categories(self):
        encoder = OneHotEncoder()

        encoder.partial_fit([["red"], ["blue"]])
        first_width = encoder.transform([["red"]]).shape[1]
        encoder.partial_fit([["green"]])

        self.assertEqual(first_width + 1, encoder.transform([["red"]]).shape[1])


if __name__ == "__main__":
    unittest.main()
