"""Sphinx configuration for the GPN documentation."""

from importlib.metadata import version as distribution_version

project = "GPN"
author = "Gonzalo Benegas, Chengzhong Ye, and contributors"
copyright = "2026, Song Lab at UC Berkeley"
release = distribution_version("gpn")
version = release

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = [
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "remove"

nitpicky = True
# External inventories are intentionally not fetched during offline documentation
# builds. Keep internal references strict while accepting external array-type names.
nitpick_ignore_regex = [
    ("py:class", r"(?:jaxtyping\.)?(?:Float|Int)(?:\[.*\])?"),
    ("py:class", r"(?:torch\.)?(?:Tensor|Module)"),
    ("py:class", r"'.*'"),
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "GPN"
html_static_path = ["_static"]
html_css_files = ["gpn.css"]
html_theme_options = {
    "repository_url": "https://github.com/songlab-cal/gpn",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    # Generated notebook pages do not have a source under docs/_notebooks.
    "use_source_button": False,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_prev_next": False,
}
