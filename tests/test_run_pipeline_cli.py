import sys

import run_pipeline


def test_rights_cleared_defaults_to_false(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "https://example.com/video"])

    args = run_pipeline.parse_args()

    assert args.rights_cleared is False
    assert args.target_language is None


def test_rights_cleared_and_target_language_can_be_set(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "https://example.com/video",
            "--rights-cleared",
            "--target-language",
            "both",
        ],
    )

    args = run_pipeline.parse_args()

    assert args.rights_cleared is True
    assert args.target_language == "both"
