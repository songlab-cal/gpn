from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_autodoc_directives_render_generated_rest_as_rest():
    api_source = (ROOT / "docs" / "reference" / "python-api.md").read_text()

    assert "```{eval-rst}" in api_source
    assert "```{autofunction}" not in api_source
    assert "```{autoclass}" not in api_source


def test_hosted_docs_install_inference_dependencies_for_api_imports():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    read_the_docs = (ROOT / ".readthedocs.yaml").read_text()

    assert "--no-default-groups --extra inference --group docs" in ci
    assert "extras:\n        - inference" in read_the_docs


def test_public_docs_are_attributed_to_song_lab():
    sphinx_config = (ROOT / "docs" / "conf.py").read_text()

    assert 'author = "Song Lab at UC Berkeley"' in sphinx_config
    assert "By Gonzalo Benegas" not in sphinx_config
