"""Pruebas de la lógica de ML (pipeline, reglas de dominio, métricas, validación).

Se usan solo unittest (stdlib) y los modulos del proyecto: no añade dependencias.
Ejecutar desde la raiz del repo:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    DomainImputer,
    QuantileClipper,
    RareCategoryGrouper,
    build_prediction_frame_from_dict,
    build_preprocessor,
    get_training_frame,
    load_features,
)
from src.train import calculate_metrics  # noqa: E402

TRAIN = ROOT / "data" / "raw" / "train.csv"


class TestManifest(unittest.TestCase):
    def test_22_features(self):
        self.assertEqual(len(load_features()), 22)

    def test_prediction_frame_shape_and_order(self):
        feats = load_features()
        df = build_prediction_frame_from_dict({"GrLivArea": 1500, "Neighborhood": "NAmes"})
        self.assertEqual(list(df.columns), feats)
        self.assertEqual(df.shape, (1, 22))


class TestDomainImputer(unittest.TestCase):
    def test_basement_na_is_structural(self):
        X = pd.DataFrame(
            {"BsmtQual": [np.nan, "Gd"], "TotalBsmtSF": [np.nan, 800.0], "GarageArea": [0.0, 400.0]}
        )
        out = DomainImputer().fit_transform(X)
        self.assertEqual(out.loc[0, "BsmtQual"], "NoBasement")
        self.assertEqual(out.loc[0, "TotalBsmtSF"], 0)
        self.assertEqual(out.loc[0, "has_bsmt"], 0)
        self.assertEqual(out.loc[1, "has_bsmt"], 1)

    def test_garage_flag(self):
        X = pd.DataFrame({"GarageArea": [0.0, 350.0], "TotalBsmtSF": [500.0, 500.0]})
        out = DomainImputer().fit_transform(X)
        self.assertEqual(out.loc[0, "has_garage"], 0)
        self.assertEqual(out.loc[1, "has_garage"], 1)

    def test_no_dead_has_masonry_flag(self):
        X = pd.DataFrame({"GarageArea": [100.0], "TotalBsmtSF": [500.0]})
        out = DomainImputer().fit_transform(X)
        self.assertNotIn("has_masonry", out.columns)


class TestQuantileClipperNoLeakage(unittest.TestCase):
    def test_bounds_learned_only_on_fit(self):
        train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        qc = QuantileClipper(columns=["x"], lower_q=0.0, upper_q=0.9).fit(train)
        _, hi = qc.bounds_["x"]
        out = qc.transform(pd.DataFrame({"x": [999]}))
        # Un valor de test por encima del limite ajustado en train se recorta a 'hi'.
        self.assertEqual(out.loc[0, "x"], hi)


class TestRareCategoryGrouper(unittest.TestCase):
    def test_rare_to_other(self):
        train = pd.DataFrame({"c": ["A"] * 198 + ["B"] + ["Z"]})  # B y Z = 0.5% < 1%
        rg = RareCategoryGrouper(columns=["c"], min_freq=0.01).fit(train)
        out = rg.transform(pd.DataFrame({"c": ["A", "Z", "B"]}))
        self.assertEqual(out.loc[0, "c"], "A")
        self.assertEqual(out.loc[1, "c"], "Other")
        self.assertEqual(out.loc[2, "c"], "Other")


class TestPreprocessorPipeline(unittest.TestCase):
    @unittest.skipUnless(TRAIN.exists(), "requiere data/raw/train.csv")
    def test_fit_transform_no_masonry_and_consistent(self):
        df = pd.read_csv(TRAIN)
        X, y = get_training_frame(df)
        pre = build_preprocessor()
        Xt = pre.fit_transform(X, y)
        names = pre.named_steps["preprocessor"].get_feature_names_out()
        self.assertEqual(Xt.shape[0], len(X))
        self.assertEqual(Xt.shape[1], len(names))
        self.assertFalse(any("has_masonry" in n for n in names), "feature muerta no debe existir")
        self.assertTrue(any("has_garage" in n for n in names))


class TestMetrics(unittest.TestCase):
    def test_rmse_is_sqrt_of_mse(self):
        y = pd.Series([100.0, 200.0, 300.0])
        p = pd.Series([110.0, 190.0, 310.0])
        m = calculate_metrics(y, p)
        self.assertAlmostEqual(m["RMSE"], m["MSE"] ** 0.5, places=6)
        self.assertGreater(m["R2"], 0.9)


class TestInputValidator(unittest.TestCase):
    """La capa de validacion de la app (se omite si streamlit no esta disponible)."""

    def test_sanitize_coerces_and_defaults(self):
        try:
            from app.validation import InputValidator
        except Exception as exc:  # streamlit u otra dep ausente
            self.skipTest(f"app.validation no importable: {exc}")

        features = ["GrLivArea", "Neighborhood"]
        defaults = {"GrLivArea": 1500.0, "Neighborhood": "NAmes"}
        values = {"GrLivArea": "no-numerico", "Neighborhood": ""}
        sanitized, notes = InputValidator.sanitize_values(values, features, None, defaults)
        self.assertEqual(sanitized["GrLivArea"], 1500.0)  # fallback numerico
        self.assertEqual(sanitized["Neighborhood"], "NAmes")  # fallback categorico
        self.assertTrue(len(notes) >= 1)


if __name__ == "__main__":
    unittest.main()
