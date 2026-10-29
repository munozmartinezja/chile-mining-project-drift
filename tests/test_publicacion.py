import csv
import re
from pathlib import Path

import pytest

from cpdrift.publicacion import exportar_registro, render

ROOT = Path(__file__).parents[1]
PLANTILLAS = (
    "README.md",
    "brief.md",
    "metodo.md",
    "preguntas_dificiles.md",
)
MARCADOR = re.compile(r"\{\{([^|}]+)\|([^}]+)\}\}")
CABECERA = "<!-- Generado desde docs/plantillas con cpdrift.publicacion. No editar a mano. -->\n"


def _registro(path: Path = ROOT / "docs/registro_cifras.csv") -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as source:
        return {row["id"]: float(row["valor"]) for row in csv.DictReader(source)}


def _renderizar_plantilla(name: str, registro: dict[str, float]) -> str:
    template = (ROOT / "docs/plantillas" / name).read_text(encoding="utf-8")
    if name == "README.md":
        spanish, english = template.split("## English summary", 1)
        body = render(spanish, registro, "es") + "## English summary" + render(
            english, registro, "en"
        )
    else:
        body = render(template, registro, "es")
    return CABECERA + body


def test_documentos_generados_estan_al_dia() -> None:
    registro = _registro()
    for name in PLANTILLAS:
        destination = ROOT / (name if name == "README.md" else f"docs/{name}")
        assert destination.read_text(encoding="utf-8") == _renderizar_plantilla(
            name, registro
        )


def test_todos_los_marcadores_son_validos() -> None:
    registro = _registro()
    for name in PLANTILLAS:
        text = (ROOT / "docs/plantillas" / name).read_text(encoding="utf-8")
        for marker in MARCADOR.finditer(text):
            claim_id, _format = marker.groups()
            assert claim_id in registro, f"{name}: id desconocido {claim_id}"
            render(marker.group(), registro, "es")


def test_plantillas_no_contienen_numeros_sueltos() -> None:
    number = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
    for name in PLANTILLAS:
        path = ROOT / "docs/plantillas" / name
        text = MARCADOR.sub("", path.read_text(encoding="utf-8"))
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\]\([^)]*\)", "]", text)
        text = text.replace("SHA-256", "").replace("Python 3.12", "")
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in number.finditer(line):
                value = match.group()
                valid = (
                    (value.isdigit() and 2000 <= int(value) <= 2039)
                    or (value.isdigit() and len(value) == 1)
                    or value in {"10", "90", "95", "100"}
                )
                assert valid, f"{path}:{line_number}: número suelto {value}"


@pytest.mark.parametrize(
    ("value", "format_name", "spanish", "english"),
    [
        (133, "ent", "133", "133"),
        (0.66, "d1", "0,7", "0.7"),
        (0.66, "d2", "0,66", "0.66"),
        (0.6666, "d3", "0,667", "0.667"),
        (50.1, "pct0", "50 %", "50%"),
        (50.1, "pct1", "50,1 %", "50.1%"),
        (0.73, "prop0", "73 %", "73%"),
        (0.06, "pp0", "6 pp", "6 pp"),
        (0.73, "de10", "7", "7"),
        (0.0009, "p", "< 0,001", "< 0.001"),
        (0.125, "p", "= 0,125", "= 0.125"),
        (0.125, "d2", "0,13", "0.13"),
    ],
)
def test_formatos(
    value: float, format_name: str, spanish: str, english: str
) -> None:
    marker = "{{valor|" + format_name + "}}"
    assert render(marker, {"valor": value}, "es") == spanish
    assert render(marker, {"valor": value}, "en") == english


def test_registro_publicado_sincronizado() -> None:
    source = ROOT / "data/powerbi/registro_cifras.csv"
    if not source.exists():
        pytest.skip("no existe data/powerbi/registro_cifras.csv")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        expected = {
            row["id"]: row
            for row in csv.DictReader(handle)
            if row["estado"] == "verificada"
        }
    with (ROOT / "docs/registro_cifras.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        published = {row["id"]: row for row in csv.DictReader(handle)}
    assert published.keys() == expected.keys()
    for claim_id, row in published.items():
        assert row["descripcion"] == expected[claim_id]["descripcion"]
        assert row["unidad"] == expected[claim_id]["unidad"]
        assert float(row["valor"]) == pytest.approx(
            float(expected[claim_id]["valor"]), abs=1e-12
        )


def test_exportar_registro_filtra_informativas_y_rechaza_estados(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    source.write_text(
        "id,descripcion,valor,unidad,estado\n"
        "a,Primera,1.5,cosas,verificada\n"
        "b,Segunda,2.5,cosas,informativa\n",
        encoding="utf-8",
    )
    exportar_registro(source, destination)
    assert destination.read_text(encoding="utf-8") == (
        "id,descripcion,valor,unidad\na,Primera,1.5,cosas\n"
    )

    source.write_text(
        "id,descripcion,valor,unidad,estado\na,Primera,1,cosas,fallida\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="estado"):
        exportar_registro(source, destination)


def test_render_rechaza_id_y_formato_desconocidos() -> None:
    with pytest.raises(ValueError, match="id"):
        render("{{desconocido|ent}}", {}, "es")
    with pytest.raises(ValueError, match="formato"):
        render("{{valor|otro}}", {"valor": 1}, "es")
