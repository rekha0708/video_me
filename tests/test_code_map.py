"""Freshness guard for docs/code_map/.

Regenerates the code map into a temp dir and asserts it matches the committed
docs/code_map/ byte-for-byte. If this fails, the code changed without the map:

    python -m scripts.generate_code_map
"""

from pathlib import Path

from scripts.generate_code_map import CURATED_FILES, DEFAULT_OUTPUT_DIR, generate

STALE_HINT = "docs/code_map/ is stale. Run: python -m scripts.generate_code_map"


def test_code_map_is_fresh(tmp_path: Path) -> None:
    pages = generate(tmp_path)
    assert pages, "generator produced no pages"
    for name, expected in pages.items():
        committed = DEFAULT_OUTPUT_DIR / name
        assert committed.exists(), f"missing docs/code_map/{name}. {STALE_HINT}"
        actual = committed.read_text(encoding="utf-8")
        assert actual == expected, f"docs/code_map/{name} differs. {STALE_HINT}"


def test_code_map_has_no_orphan_files(tmp_path: Path) -> None:
    """Every file in docs/code_map/ is either generated or explicitly curated."""
    generated_names = set(generate(tmp_path).keys())
    for path in sorted(DEFAULT_OUTPUT_DIR.iterdir()):
        assert path.name in generated_names or path.name in CURATED_FILES, (
            f"unexpected file in docs/code_map/: {path.name}. "
            "Add it to CURATED_FILES in scripts/generate_code_map.py or remove it."
        )


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first = generate(tmp_path / "a")
    second = generate(tmp_path / "b")
    assert first == second
