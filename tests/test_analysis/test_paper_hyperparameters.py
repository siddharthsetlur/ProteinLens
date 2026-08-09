from proteinlens.analysis.geometry.classifiers import paper_gbm_parameters


def test_appendix_table_7_gbm_parameters():
    assert paper_gbm_parameters(1_000) == {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
        "min_samples_leaf": 20,
    }


def test_appendix_table_7_leaf_floor():
    assert paper_gbm_parameters(100)["min_samples_leaf"] == 5
