from __future__ import annotations

import datetime
import sys
import typing

import sphinx.util.logging

import sa_preisach

if typing.TYPE_CHECKING:
    import sphinx.application


project = "sa-preisach"
author = "Anton Lu"
version = sa_preisach.__version__

copyright = f"{datetime.datetime.now().year}, CERN"  # noqa: A001


# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "acc_py_sphinx.theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_parser",  # markdown support
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "acc_py"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_show_sphinx = False
html_show_sourcelink = True


# -- Options for sphinx.ext.autosummary

autosummary_generate = True
autosummary_imported_members = True

autosectionlabel_prefix_document = True

autoclass_content = "init"
autodoc_typehints = "both"
autodoc_default_options = {
    "show-inheritance": True,
}

# -- Options for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
}


logger = sphinx.util.logging.getLogger(__name__)


def hijack_module_name_replacement() -> None:
    # Working in sphinx environment, not replacing due to
    # https://github.com/sphinx-doc/sphinx/issues/11031
    logger.info("Disabling __module__ rewrite")
    import sa_preisach._mod_replace

    for mod_name in list(sys.modules):
        if mod_name == "sa_preisach._mod_replace":
            continue
        if mod_name == "sa_preisach" or mod_name.startswith("sa_preisach."):
            del sys.modules[mod_name]

    sa_preisach._mod_replace.replace_modname = lambda *_: None  # noqa: SLF001


hijack_module_name_replacement()


def remove__init__from_docs(
    app: sphinx.application.Sphinx,
    what: str,
    name: str,
    obj: object,
    skip: bool,  # noqa: FBT001
    options: dict,
) -> bool | None:
    # Skip the __init__ methods of all classes.
    # Reminder: a handler must return True to skip, None to delegate. False means that
    # the thing MUST not be skipped (giving no chance for downstream handlers to influence
    # the skipping behaviour).

    # Unused args:
    _ = app, what, obj, skip, options

    if name == "__init__":
        # Skip __init__, which must be documented in the class docstring.
        return True

    return None


nitpicky = True


def setup(app: sphinx.application.Sphinx) -> None:
    app.connect("autodoc-skip-member", remove__init__from_docs)
