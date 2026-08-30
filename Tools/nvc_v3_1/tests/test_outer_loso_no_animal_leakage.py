from ..validation import run_outer_loso


def test_outer_loso_no_animal_leakage(synthetic_features):
    _, audit, _ = run_outer_loso(synthetic_features, "PE", 0.5)
    for row in audit.itertuples(index=False):
        assert row.outer_held_out_animal not in row.outer_training_animals.split("+")
        assert row.leakage is False
