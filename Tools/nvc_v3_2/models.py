"""Small, preregistered classifiers for V3.2."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .version_support import expanded_animal_class_weights
from . import config as C


def assert_feature_schema_safe(names):
    offenders = []
    for name in names:
        low = str(name).casefold()
        if low in C.FORBIDDEN_FEATURE_TOKENS or any(t in low for t in ("urine", "volume", "future", "recovery", "outer_fold")):
            offenders.append(str(name))
    if offenders:
        raise AssertionError(f"evaluation/identity fields in feature schema: {offenders}")


def animal_equal_weights(frame, target_col="target"):
    x = frame.copy(); subjects = x["subject"].astype(str)
    counts = x.groupby([subjects, target_col]).size().to_dict()
    animals = sorted(subjects.unique())
    if not animals or any(not any((a, t) in counts for t in (0, 1)) for a in animals):
        # Inner folds can lack one class in a rare animal; equal-animal weighting remains valid.
        return np.ones(len(x), dtype=float)
    classes = {a: [t for t in (0, 1) if (a, t) in counts] for a in animals}
    return np.asarray([1.0 / (len(animals) * len(classes[a]) * counts[(a, int(t))])
                       for a, t in zip(subjects, x[target_col])], dtype=float)


def fit_classifier(frame, features, kind="lr", C_value=1.0, l1_ratio=0.5,
                    target_col="target", allowed_subjects=()):
    features = tuple(features); assert_feature_schema_safe(features)
    d = frame[frame.get("model_scorable", True).astype(bool)].copy() if hasattr(frame.get("model_scorable", True), "astype") else frame.copy()
    d = d[np.isfinite(d[list(features)].to_numpy(float)).all(axis=1)]
    if allowed_subjects:
        if set(d["subject"].astype(str)) - set(map(str, allowed_subjects)):
            raise AssertionError("held-out animal entered fit")
    y = d[target_col].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("training fold does not contain both classes")
    if kind == "svm":
        estimator = LinearSVC(C=float(C_value), class_weight=None, max_iter=5000, random_state=C.RANDOM_STATE)
        model = Pipeline([("scaler", StandardScaler()), ("svm", estimator)])
    else:
        penalty = "elasticnet" if kind == "elasticnet" else "l2"
        solver = "saga" if penalty == "elasticnet" else "lbfgs"
        estimator = LogisticRegression(C=float(C_value), penalty=penalty, l1_ratio=float(l1_ratio) if penalty == "elasticnet" else None,
                                       solver=solver, max_iter=600 if penalty == "elasticnet" else 2000,
                                       tol=1e-3 if penalty == "elasticnet" else 1e-4,
                                       n_jobs=-1 if penalty == "elasticnet" else None,
                                       random_state=C.RANDOM_STATE)
        model = Pipeline([("scaler", StandardScaler()), ("logistic", estimator)])
    model.fit(d[list(features)], y, **({"svm__sample_weight": animal_equal_weights(d, target_col)} if kind == "svm" else {"logistic__sample_weight": animal_equal_weights(d, target_col)}))
    model.fit_subjects_ = tuple(sorted(map(str, allowed_subjects)))
    model.fit_features_ = features
    model.kind_ = kind
    model.target_col_ = target_col
    model.hyperparameters_ = {"C": float(C_value), "l1_ratio": float(l1_ratio) if kind == "elasticnet" else None}
    return model


def model_scores(model, frame):
    x = frame[list(model.fit_features_)]
    if model.kind_ == "svm":
        return model.decision_function(x).astype(float)
    return model.predict_proba(x)[:, list(model.named_steps["logistic"].classes_).index(1)].astype(float)
