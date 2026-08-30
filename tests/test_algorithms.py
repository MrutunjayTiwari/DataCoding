import numpy as np

from datacoding.algorithms import (
    PCA,
    KMeans,
    KNNClassifier,
    LinearRegressionGD,
    LogisticRegressionGD,
    PerceptronClassifier,
)


def test_linear_regression_recovers_simple_relationship():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.25
    model = LinearRegressionGD(learning_rate=0.08, max_iter=3_000).fit(X, y)
    mse = np.mean((model.predict(X) - y) ** 2)
    assert mse < 1e-8
    assert np.allclose(model.coef_, [2.0, -3.0], atol=1e-4)
    assert np.isclose(model.intercept_, 1.25, atol=1e-4)


def test_logistic_regression_separates_two_blobs():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(-2, 0.5, size=(100, 2)), rng.normal(2, 0.5, size=(100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    model = LogisticRegressionGD(learning_rate=0.2, max_iter=2_000).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.99
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_perceptron_converges_on_linearly_separable_data():
    X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], dtype=float)
    y = np.array([-1, -1, 1, 1])
    model = PerceptronClassifier(max_epochs=20).fit(X, y)
    assert np.array_equal(model.predict(X), y)
    assert model.mistakes_per_epoch_[-1] == 0


def test_knn_uses_majority_vote():
    X = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]], dtype=float)
    y = np.array(["low"] * 3 + ["high"] * 3)
    model = KNNClassifier(n_neighbors=3).fit(X, y)
    assert model.predict([[0.2, 0.1], [9.4, 9.2]]).tolist() == ["low", "high"]


def test_kmeans_finds_three_centers():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [
            rng.normal([0, 0], 0.15, size=(80, 2)),
            rng.normal([5, 0], 0.15, size=(80, 2)),
            rng.normal([2.5, 4], 0.15, size=(80, 2)),
        ]
    )
    model = KMeans(n_clusters=3, random_state=4).fit(X)
    assert len(np.unique(model.labels_)) == 3
    assert model.inertia_ < 20


def test_pca_round_trip_and_variance_ratio():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 3))
    X[:, 0] *= 8
    model = PCA(n_components=2)
    reduced = model.fit_transform(X)
    reconstructed = model.inverse_transform(reduced)
    assert reduced.shape == (300, 2)
    assert model.explained_variance_ratio_.sum() > 0.95
    assert np.mean((X - reconstructed) ** 2) < 1.0
