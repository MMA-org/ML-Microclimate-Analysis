# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from sphinx.ext.intersphinx import fetch_inventory
import os
import sys
import re
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx.ext.intersphinx


project = 'ML-Microclimate-Analysis'
copyright = '2025, Nave Cohen, Erik Pinhasov'
author = 'Nave Cohen, Erik Pinhasov'

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../src")))
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Automatically document code
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',
    "myst_parser",  # Link to source code
    "sphinx.ext.doctest",
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx_copybutton',
    "sphinx.ext.ifconfig",
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/main/en/', None),
    'albumentations': ('https://albumentations.ai/docs/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pillow': ('https://pillow.readthedocs.io/en/stable/', None),
}

source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
# html_theme_options = {
#    "light_logo": "black-logo.png",
#    "dark_logo": "black-logo.png",
# }

autodoc_typehints = "description"  # Include type hints in descriptions
napoleon_use_ivar = True  # Use 'ivar' for instance variables
napoleon_use_param = True  # Document function arguments
napoleon_use_rtype = True  # Do not separate return type


# Strips Python prompts (>>> and ...) and shell prompt ($)
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_only_copy_prompt_lines = False
pygments_style = "sphinx"

autodoc_default_options = {
    'members': True,  # Do not show class and function members by default
    'undoc-members': True,  # Do not show undocumented members
    'show-inheritance': True,
}


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """
    Process docstrings to replace short references with fully qualified paths,
    avoiding replacements if preceded by a dot or followed by parentheses.
    """
    # Define short replacements
    replacements = {
        r"\bnp\.ndarray\b": ":class:`~numpy.ndarray`",
        r"\bnumpy\.ndarray\b": ":class:`~numpy.ndarray`",
        r"\bTensor\b": ":class:`~torch.Tensor`",
        r"\btorch.Tensor\b": ":class:`~torch.Tensor`",
        r"\bPIL.Image.Image\b": ":class:`~PIL.Image.Image`",
        r"\bImage.Image\b": ":class:`~PIL.Image.Image`",
        r"\bImage\b": ":class:`~PIL.Image.Image`",
        r"\bAdam\b": ":class:`~torch.optim.Adam`",
        r"\bnp.ndarray\b": ":class:`~numpy.ndarray`",
        r"\bSGD\b": ":class:`~torch.optim.SGD`",
        r"\bReduceLROnPlateau\b": ":class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`",
        r"\bLambdaLR\b": ":class:`~torch.optim.lr_scheduler.LambdaLR`",
        r"\bSegformerForSemanticSegmentation\b": ":class:`~transformers.SegformerForSemanticSegmentation`",
        r"\bSegMetrics\b": ":class:`~utils.metrics.SegMetrics`",
        r"\bTestMetrics\b": ":class:`~utils.metrics.TestMetrics`",
        r"\bFocalLoss\b": ":class:`~utils.metrics.FocalLoss`",
        r"\bConfig\b": ":class:`~utils.config.Config`",
        r"\bCheckpointDirectoryError\b": "`~core.errors.CheckpointDirectoryError`",
        r"\bCheckpointNotFoundError\b": "`~core.errors.CheckpointNotFoundError`",
        r"\bMultipleCheckpointsError\b": "`~core.errors.MultipleCheckpointsError`",
        r"\bConfigId2LabelError\b": "`~core.errors.ConfigId2LabelError`",
        r"\bFocalAlphaTypeError\b": "`~core.errors.FocalAlphaTypeError`",
        r"\bFocalAlphaSizeError\b": "`~core.errors.FocalAlphaSizeError`",
        r"\bNormalizeError\b": "`~core.errors.NormalizeError`",
        r"\bPath\b": ":class:`~pathlib.Path`",
        r"\bint\b": ":class:`int`",
        r"\bfloat\b": ":class:`float`",
        r"\bstr\b": ":class:`str`",
        r"\blist\b": ":class:`list`",
        r"\bdict\b": ":class:`dict`",
    }

    for i, line in enumerate(lines):
        original_line = line

        # Apply replacements only if the term is not preceded by a dot or followed by parentheses
        for pattern, replacement in replacements.items():
            if re.search(pattern, line):
                # Skip replacement if preceded by a dot (e.g., `torch.Tensor` or `utils.metrics.FocalLoss`)
                if re.search(r"\.\b" + pattern[2:], line):
                    continue
                # Skip replacement if followed by parentheses (e.g., `FocalLoss(...)`)
                if re.search(pattern + r"\s*\(", line):
                    continue
                # Perform the replacement
                line = re.sub(pattern, replacement, line)
        # Update the line in the list
        lines[i] = line


def setup(app):
    """
    Connects the autodoc-process-docstring event to the custom processing function.
    """
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
