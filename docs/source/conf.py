import os
import sys
from sphinx.ext.autodoc import ClassDocumenter, _
import re
# -- Project information -----------------------------------------------------
project = 'ML-Microclimate-Analysis'
copyright = '2025, Nave Cohen, Erik Pinhasov'
author = 'Nave Cohen, Erik Pinhasov'

# -- Setup paths -------------------------------------------------------------
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../src")))

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',         # Automatically document code
    'sphinx.ext.napoleon',        # Support for Google-style docstrings
    'sphinx.ext.viewcode',        # Add links to source code
    'myst_parser',                # Markdown support
    'sphinx.ext.doctest',         # Run doctests
    'sphinx.ext.intersphinx',     # Link to other project's documentation
    'sphinx_copybutton',          # Add copy buttons to code blocks
    'sphinx.ext.ifconfig',        # Conditional content
]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/main/en/', None),
    'albumentations': ('https://albumentations.ai/docs/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'datasets': ('https://huggingface.co/docs/datasets/main/en/', None),
}

# -- Source configuration ----------------------------------------------------
source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': False,  # Exclude undocumented members
    'exclude-members': '__weakref__',  # Exclude specific members
    'show-inheritance': True,
}

autodoc_typehints = "description"  # Include type hints in descriptions
autodoc_typehints_format = "short"  # Use short names for type hints
# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Copybutton configuration ------------------------------------------------
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_only_copy_prompt_lines = False

# -- Pygments configuration --------------------------------------------------
pygments_style = "sphinx"

# -- Custom modifications to Sphinx behavior ---------------------------------
# Remove 'Bases: object' from class docstrings
add_line = ClassDocumenter.add_line
line_to_delete = _('Bases: %s') % ':py:class:`object`'


def add_line_no_object_base(self, text, *args, **kwargs):
    if text.strip() == line_to_delete:
        return
    add_line(self, text, *args, **kwargs)


add_directive_header = ClassDocumenter.add_directive_header


def add_directive_header_no_object_base(self, *args, **kwargs):
    self.add_line = add_line_no_object_base.__get__(self)
    result = add_directive_header(self, *args, **kwargs)
    del self.add_line
    return result


ClassDocumenter.add_directive_header = add_directive_header_no_object_base

replacements = {
    r"\bnp\.ndarray\b": ":class:`~numpy.ndarray`",
    r"\btorch\.Tensor\b": ":class:`~torch.Tensor`",
    r"\bDataLoader\b": ":class:`~torch.utils.data.DataLoader`",
    r"\btorch\.nn\.CrossEntropyLoss\b": ":class:`~torch.nn.CrossEntropyLoss`",
    r"\bLightningDataModule\b": ":class:`~pytorch_lightning.LightningDataModule`",
    r"\bpl\.LightningModule\b": ":class:`~pytorch_lightning.LightningModule`",
    r"\bPIL.Image.Image\b": ":class:`~PIL.Image.Image`",
    r"\bImage.Image\b": ":class:`~PIL.Image.Image`",
    r"\bImage\b": ":class:`~PIL.Image.Image`",
    r"\bAdam\b": ":class:`~torch.optim.Adam`",
    r"\bAdamW\b": ":class:`~torch.optim.AdamW`",
    r"\bSGD\b": ":class:`~torch.optim.SGD`",
    r"\bReduceLROnPlateau\b": ":class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`",
    r"\bLambdaLR\b": ":class:`~torch.optim.lr_scheduler.LambdaLR`",
    r"\bSegformerForSemanticSegmentation\b": ":class:`~transformers.SegformerForSemanticSegmentation`",
    r"\bSegformerImageProcessor\b": ":class:`~transformers.SegformerImageProcessor`",
    r"\balbumentations\.Compose\b": ":class:`~albumentations.Compose`",
    r"\bDataset\b": ":class:`~datasets.Dataset`",
    r"\bSegMetrics\b": ":class:`~utils.metrics.SegMetrics`",
    r"\bTestMetrics\b": ":class:`~utils.metrics.TestMetrics`",
    r"\bFocalLoss\b": ":class:`~utils.metrics.FocalLoss`",
    r"\bCeDiceLoss\b": ":class:`~utils.metrics.CeDiceLoss`",
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
    r"\bbool\b": ":class:`bool`",
}


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """
    Process docstrings to replace short references with fully qualified paths,
    avoiding replacements if preceded by a dot or followed by parentheses.
    """

    for i, line in enumerate(lines):
        # Apply replacements only if the term is not preceded by a dot or followed by parentheses
        for pattern, replacement in replacements.items():
            # Ensure we only replace the exact class name and not a part of another word (e.g., `torch.Tensor` or `utils.metrics.FocalLoss`)
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
