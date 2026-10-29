"""Render the verified publication register into repository documentation."""

from __future__ import annotations

import csv
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

MARKER = re.compile(r"\{\{([^|}]+)\|([^}]+)\}\}")
FORMATS = {"ent", "d1", "d2", "d3", "pct0", "pct1", "prop0", "pp0", "de10", "p"}
HEADER = "<!-- Generado desde docs/plantillas con cpdrift.publicacion. No editar a mano. -->\n"


def exportar_registro(origen: Path, destino: Path) -> None:
    """Publish verified rows from the pipeline register without a UTF-8 BOM."""
    with origen.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    invalid = sorted({row["estado"] for row in rows} - {"verificada", "informativa"})
    if invalid:
        raise ValueError(f"estado no publicable: {', '.join(invalid)}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["id", "descripcion", "valor", "unidad"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames}
            for row in rows
            if row["estado"] == "verificada"
        )


def _fixed(value: Decimal, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    return f"{value.quantize(quantum, rounding=ROUND_HALF_UP):.{decimals}f}"


def render(texto: str, registro: dict[str, float], idioma: str) -> str:
    """Replace publication markers using locale-aware, half-up formatting."""
    if idioma not in {"es", "en"}:
        raise ValueError(f"idioma desconocido: {idioma}")

    def replace(match: re.Match[str]) -> str:
        claim_id, format_name = match.groups()
        if claim_id not in registro:
            raise ValueError(f"id desconocido: {claim_id}")
        if format_name not in FORMATS:
            raise ValueError(f"formato desconocido: {format_name}")
        value = Decimal(str(registro[claim_id]))
        suffix = ""
        if format_name == "ent":
            result = _fixed(value, 0)
        elif format_name in {"d1", "d2", "d3"}:
            result = _fixed(value, int(format_name[1]))
        elif format_name.startswith("pct"):
            result = _fixed(value, int(format_name[-1]))
            suffix = " %" if idioma == "es" else "%"
        elif format_name == "prop0":
            result = _fixed(value * 100, 0)
            suffix = " %" if idioma == "es" else "%"
        elif format_name == "pp0":
            result = _fixed(value * 100, 0)
            suffix = " pp"
        elif format_name == "de10":
            result = _fixed(value * 10, 0)
        elif value < Decimal("0.001"):
            result = "< 0.001"
        else:
            result = f"= {_fixed(value, 3)}"
        if idioma == "es":
            result = result.replace(".", ",")
        return result + suffix

    return MARKER.sub(replace, texto)


def publicar(raiz: Path) -> None:
    """Publish the verified register and render all repository templates."""
    published_register = raiz / "docs/registro_cifras.csv"
    exportar_registro(raiz / "data/powerbi/registro_cifras.csv", published_register)
    with published_register.open(encoding="utf-8", newline="") as source:
        register = {
            row["id"]: float(row["valor"]) for row in csv.DictReader(source)
        }

    for name in ("README.md", "brief.md", "metodo.md", "preguntas_dificiles.md"):
        template = (raiz / "docs/plantillas" / name).read_text(encoding="utf-8")
        if name == "README.md":
            spanish, separator, english = template.partition("## English summary")
            if not separator:
                raise ValueError("README.md no contiene ## English summary")
            body = render(spanish, register, "es") + render(
                separator + english, register, "en"
            )
            destination = raiz / name
        else:
            body = render(template, register, "es")
            destination = raiz / "docs" / name
        destination.write_text(HEADER + body, encoding="utf-8")


if __name__ == "__main__":
    publicar(Path("."))
