from pathlib import Path
import sys
import joblib
import pandas as pd

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.preprocessing import get_training_frame, build_preprocessor, load_features


def main():
	raw_path = ROOT / "data" / "raw" / "train.csv"
	processed_dir = ROOT / "data" / "processed"
	models_dir = ROOT / "models"
	processed_dir.mkdir(parents=True, exist_ok=True)
	models_dir.mkdir(parents=True, exist_ok=True)

	print(f"Cargando datos desde: {raw_path}")
	df = pd.read_csv(raw_path)

	print("Extrayendo X,y con el manifest de features...")
	X, y = get_training_frame(df)
	print(f"X shape: {X.shape}, y shape: {y.shape}")

	print("Construyendo y ajustando el preprocessor (DomainImputer + ColumnTransformer)...")
	preprocessor = build_preprocessor()
	preprocessor.fit(X)

	print("Transformando datos de entrenamiento...")
	X_trans = preprocessor.transform(X)

	# intentar obtener nombres de columnas, si sklearn soporta get_feature_names_out
	try:
		ct = preprocessor.named_steps["preprocessor"]
		feature_names = ct.get_feature_names_out()
	except Exception:
		feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

	out_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)

	out_csv = processed_dir / "train_preprocessed.csv"
	print(f"Guardando preprocessed CSV en: {out_csv}")
	out_df.to_csv(out_csv, index=True)

	preproc_file = models_dir / "preprocessor.joblib"
	print(f"Guardando preprocessor ajustado en: {preproc_file}")
	joblib.dump(preprocessor, preproc_file)

	print("Hecho.")


if __name__ == '__main__':
	main()

