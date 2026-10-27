from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from cpdrift.fuentes import descargar_fuentes, verificar_fuentes


def _manifest(path: Path, content: bytes, *, actualizable: str = "no") -> None:
    pd.DataFrame(
        [{
            "archivo": "fuente.bin",
            "url": "https://invalid.example/fuente.bin",
            "sha256": hashlib.sha256(content).hexdigest(),
            "actualizable": actualizable,
            "nota": "prueba",
        }]
    ).to_csv(path, index=False)


def test_download_uses_injected_fetch_and_verifies(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.csv"
    expected = b"contenido fijo"
    _manifest(manifest, expected)
    calls = []

    def fetch(url: str, path: Path) -> None:
        calls.append(url)
        path.write_bytes(expected)

    assert descargar_fuentes(raw, manifest, fetch=fetch) == []
    assert calls == ["https://invalid.example/fuente.bin"]


def test_fixed_source_hash_mismatch_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fuente.bin").write_bytes(b"distinto")
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, b"esperado")

    with pytest.raises(ValueError, match="SHA-256"):
        verificar_fuentes(raw, manifest)


def test_mutable_source_hash_mismatch_is_a_warning(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fuente.bin").write_bytes(b"distinto")
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, b"esperado", actualizable="si")

    assert verificar_fuentes(raw, manifest) == [
        "hash SHA-256 distinto para fuente.bin"
    ]
