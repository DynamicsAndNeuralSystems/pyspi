# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PySPI'
copyright = '2022, Oliver M. Cliff'
author = 'Oliver M. Cliff'

release = '0.3'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
]

napoleon_use_param = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/',None)
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

napoleon_type_aliases = {
    'array-like': ':term:`array-like <array_like>`',
    'array_like': ':term:`array_like`',
    "dict-like": ":term:`dict-like <mapping>`",
}