import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent
train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"

# Cargar los datos
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df = pd.concat([df_train, df_test], ignore_index=True)

reporte = "# Variables obtenidas\n\n"

for col in df.columns:
    valores = df[col].dropna().unique()
    tipo = df[col].dtype
    reporte += f"## {col}\n\n"
    if tipo in ["float64", "int64"]:
        if len(valores) > 15:
            reporte += "[Datos numéricos variables]\n\n"
        else:
            valores_ordenados = sorted(valores)
            reporte += ", ".join(str(v) for v in valores_ordenados) + "\n\n"
    else:
        valores_ordenados = sorted(
            valores, key=lambda x: (str(x).lower() if isinstance(x, str) else x)
        )
        reporte += ", ".join(str(v) for v in valores_ordenados) + "\n\n"

with open(base_path / "output" / "reporte_variables.md", "w", encoding="utf-8") as f:
    f.write(reporte)
