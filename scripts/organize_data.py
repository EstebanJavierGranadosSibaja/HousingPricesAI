"""Organiza los datos en `data/raw`.

Modo por defecto: `--dry-run` (no mueve archivos). Use `--move` para mover archivos.
"""
import os
import shutil
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT, 'house-prices-advanced-regression-techniques')
DEST_DIR = os.path.join(ROOT, 'data', 'raw')


def scan():
    candidates = []
    if os.path.isdir(SRC_DIR):
        for fname in os.listdir(SRC_DIR):
            if fname.lower().endswith('.csv'):
                candidates.append(os.path.join(SRC_DIR, fname))
    return candidates


def move_files(files, dest, dry_run=True):
    os.makedirs(dest, exist_ok=True)
    for f in files:
        dest_p = os.path.join(dest, os.path.basename(f))
        if dry_run:
            print('[dry-run] Mover:', f, '->', dest_p)
        else:
            print('Moviendo:', f, '->', dest_p)
            shutil.move(f, dest_p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--move', action='store_true', help='Mover archivos en lugar de simular')
    args = parser.parse_args()

    files = scan()
    if not files:
        print('No se encontraron CSVs en', SRC_DIR)
    else:
        print('Encontrados:', files)
        move_files(files, DEST_DIR, dry_run=not args.move)
        if not args.move:
            print('\nSimulación completa. Vuelva a ejecutar con --move para mover archivos.')
