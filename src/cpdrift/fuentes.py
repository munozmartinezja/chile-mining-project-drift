"""Download and cryptographically verify the local source manifest."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["archivo", "url", "sha256", "actualizable", "nota"]


def _manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if table.columns.tolist() != MANIFEST_COLUMNS:
        raise ValueError(f"columnas inválidas en manifiesto de fuentes: {path}")
    if not table["actualizable"].isin({"si", "no"}).all():
        raise ValueError("actualizable debe ser 'si' o 'no'")
    return table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verificar_fuentes(raw_dir: Path, manifest_path: Path) -> list[str]:
    """Verify every manifest entry, warning only for mutable sources."""
    warnings: list[str] = []
    failures: list[str] = []
    for row in _manifest(manifest_path).itertuples(index=False):
        path = raw_dir / row.archivo
        issue = None
        if not path.is_file():
            issue = f"falta {row.archivo}"
        else:
            observed = _sha256(path)
            if observed.casefold() != row.sha256.casefold():
                issue = f"hash SHA-256 distinto para {row.archivo}"
        if issue is None:
            continue
        if row.actualizable == "si":
            warnings.append(issue)
        else:
            failures.append(issue)
    if failures:
        raise ValueError("; ".join(failures))
    return warnings


def descargar_fuentes(
    raw_dir: Path,
    manifest_path: Path,
    fetch: Callable[[str, Path], object] | None = None,
) -> list[str]:
    """Download missing files and then verify the complete manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloader = fetch or (lambda url, path: urllib.request.urlretrieve(url, path))
    for row in _manifest(manifest_path).itertuples(index=False):
        path = raw_dir / row.archivo
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            downloader(row.url, path)
    return verificar_fuentes(raw_dir, manifest_path)


def main() -> None:
    warnings = descargar_fuentes(Path("data/raw"), Path("data/reference/fuentes_raw.csv"))
    for warning in warnings:
        print(f"AVISO: {warning}")
    print("Fuentes verificadas")


if __name__ == "__main__":
    main()
