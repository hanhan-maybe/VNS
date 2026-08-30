from ..config import SUBJECTS
from ..validation import inner_oof


def test_inner_loso_no_animal_leakage(synthetic_features):
    subjects = SUBJECTS[:-1]
    result = inner_oof(synthetic_features, "PE", 0.5, subjects)
    for row in result[["inner_held_animal", "inner_fit_animals"]].drop_duplicates().itertuples(index=False):
        assert row.inner_held_animal not in row.inner_fit_animals.split("+")
