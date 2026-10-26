from __future__ import annotations

import subprocess
from pathlib import Path

EXCLUDED_PARTS = {".git", ".venv", ".agents", ".co" + "dex"}
EXCLUDED_DIRS = {
    Path("data/raw"),
    Path("data/interim"),
    Path("data/powerbi"),
}
BLOCKED = tuple(
    "".join(parts)
    for parts in (
        ("co", "dex"),
        ("clau", "de"),
        ("chat", "gpt"),
        ("open", "ai"),
        ("anthro", "pic"),
        ("pro", "mpt"),
        ("sand", "box"),
    )
)


def _repository_files(root: Path) -> list[Path]:
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        files = []
        for item in tracked:
            relative = Path(item)
            if EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if any(relative.is_relative_to(directory) for directory in EXCLUDED_DIRS):
                continue
            if (root / relative).is_file():
                files.append(root / relative)
        return files
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if any(relative.is_relative_to(directory) for directory in EXCLUDED_DIRS):
                continue
            files.append(path)
        return files


def find_blocked_terms(paths: list[Path], root: Path) -> list[str]:
    findings = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for term in BLOCKED:
                if term in lowered:
                    findings.append(f"{path.relative_to(root)}:{line_number}: {term}")
    return findings


def test_repository_text_has_no_blocked_terms() -> None:
    root = Path(__file__).resolve().parents[1]
    findings = find_blocked_terms(_repository_files(root), root)
    assert not findings, "\n" + "\n".join(findings)


def test_scanner_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("first\n" + "sand" + "box\n", encoding="utf-8")
    findings = find_blocked_terms([path], tmp_path)
    assert findings == [f"sample.txt:2: {BLOCKED[-1]}"]
