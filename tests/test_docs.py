from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_autodoc_directives_render_generated_rest_as_rest():
    api_source = (ROOT / "docs" / "api.md").read_text()

    assert "```{eval-rst}" in api_source
    assert "```{autofunction}" not in api_source
    assert "```{autoclass}" not in api_source
