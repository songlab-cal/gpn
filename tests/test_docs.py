from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_python_api_documentation_is_not_built():
    sphinx_config = (ROOT / "docs" / "conf.py").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    docs_index = (ROOT / "docs" / "index.md").read_text()

    assert not (ROOT / "docs" / "reference" / "index.md").exists()
    assert not (ROOT / "docs" / "reference" / "python-api.md").exists()
    assert "autodoc" not in sphinx_config
    assert "sphinx-autodoc-typehints" not in project
    assert "python-api" not in docs_index


def test_hosted_docs_use_only_the_docs_dependency_group():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    read_the_docs = (ROOT / ".readthedocs.yaml").read_text()

    assert "--no-default-groups --group docs" in ci
    assert "--extra inference" not in ci
    assert "extras:" not in read_the_docs


def test_public_docs_are_attributed_to_song_lab():
    sphinx_config = (ROOT / "docs" / "conf.py").read_text()

    assert 'author = "Song Lab at UC Berkeley"' in sphinx_config
    assert "By Gonzalo Benegas" not in sphinx_config


def test_public_documentation_links_use_read_the_docs():
    readme = (ROOT / "README.md").read_text()
    project = (ROOT / "pyproject.toml").read_text()

    assert "https://gpn.readthedocs.io/" in readme
    assert "github.com/songlab-cal/gpn/blob/main/docs" not in readme
    assert 'Documentation = "https://gpn.readthedocs.io/"' in project
