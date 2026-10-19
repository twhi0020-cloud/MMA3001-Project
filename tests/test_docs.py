"""The documentation stays in sync with the code.

The rubric asks for docstrings that describe parameters, return values and exceptions, and for HTML
documentation that is complete and in sync with the code. These tests enforce that mechanically:

* every module, public function, public class and public method has a docstring;
* every parameter is described in the ``Args:`` section (class parameters in the class docstring);
* a function that returns a value has a ``Returns:`` section, and one that raises has ``Raises:``;
* dataclass fields are listed under ``Attributes:``;
* the generated HTML in ``docs/`` mentions every public function and class;
* on the pinned toolchain (Python, pdoc and the packages it renders with, as in ``requirements-lock.txt``), the committed HTML is identical to a fresh
  regeneration, so the documentation cannot be stale.

If the last one fails, regenerate the HTML (from the repository root)::

    PYTHONPATH=src python -m pdoc occupancy_forecast -o docs --docformat google
"""

import ast
import dataclasses
import importlib
import inspect
import os
import pkgutil
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import occupancy_forecast

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODULES = sorted(m.name for m in pkgutil.iter_modules(occupancy_forecast.__path__))


def _import(name):
    return importlib.import_module(f"occupancy_forecast.{name}")


def _public_functions_and_classes(mod):
    out = []
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out.append((name, obj))
    return out


def _callables_to_check():
    """(label, callable, docstring, kind) for every public function and public method."""
    items = []
    for m in MODULES:
        mod = _import(m)
        for name, obj in _public_functions_and_classes(mod):
            if inspect.isfunction(obj):
                items.append((f"{m}.{name}", obj, inspect.getdoc(obj)))
            else:
                for mname, meth in vars(obj).items():
                    if not mname.startswith("_") and inspect.isfunction(meth):
                        items.append((f"{m}.{name}.{mname}", meth, inspect.getdoc(meth)))
    return items


def _section(doc, title):
    """Text of a Google-style section (Args / Returns / Raises ...), or None if absent."""
    lines, grab, found = doc.splitlines(), False, []
    for line in lines:
        if line.strip() == f"{title}:":
            grab = True
            continue
        if grab:
            if line and not line.startswith((" ", "\t")):
                break
            found.append(line)
    return "\n".join(found) if grab else None


def _ast_facts(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    returns_value = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(func))
    raises = any(isinstance(n, ast.Raise) for n in ast.walk(func))
    return returns_value, raises


def test_every_module_has_a_docstring():
    assert occupancy_forecast.__doc__ and "pipeline" in occupancy_forecast.__doc__.lower()
    for m in MODULES:
        assert (_import(m).__doc__ or "").strip(), f"module {m} has no docstring"


@pytest.mark.parametrize("label,fn,doc", _callables_to_check(), ids=lambda v: v if isinstance(v, str) else "")
def test_function_docstring_describes_parameters_returns_and_raises(label, fn, doc):
    assert doc, f"{label} has no docstring"
    params = [p for p in inspect.signature(fn).parameters.values() if p.name not in ("self", "cls")]
    if params:
        args = _section(doc, "Args")
        assert args is not None, f"{label} has parameters but no 'Args:' section"
        for p in params:
            assert p.name in args, f"{label}: parameter '{p.name}' is not described in 'Args:'"
    returns_value, raises = _ast_facts(fn)
    if returns_value:
        assert _section(doc, "Returns") is not None or _section(doc, "Yields") is not None, \
            f"{label} returns a value but has no 'Returns:' section"
    if raises:
        assert _section(doc, "Raises") is not None, f"{label} raises but has no 'Raises:' section"


def _classes():
    return [(f"{m}.{n}", c) for m in MODULES for n, c in _public_functions_and_classes(_import(m)) if inspect.isclass(c)]


@pytest.mark.parametrize("label,cls", _classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_class_docstring_describes_constructor_arguments_or_fields(label, cls):
    doc = inspect.getdoc(cls)
    assert doc, f"class {label} has no docstring"
    if dataclasses.is_dataclass(cls):
        attrs = _section(doc, "Attributes")
        assert attrs is not None, f"dataclass {label} has no 'Attributes:' section"
        for f in dataclasses.fields(cls):
            assert f.name in attrs, f"{label}: field '{f.name}' is not listed under 'Attributes:'"
    elif "__init__" in vars(cls):
        params = [p for p in inspect.signature(cls.__init__).parameters.values() if p.name != "self"]
        if params:
            args = _section(doc, "Args")
            assert args is not None, f"class {label} takes arguments but has no 'Args:' section"
            for p in params:
                assert p.name in args, f"{label}: constructor argument '{p.name}' is not described in 'Args:'"


@pytest.mark.skipif(not DOCS.exists(), reason="docs/ has not been generated")
def test_generated_html_mentions_every_public_function_and_class():
    assert (DOCS / "occupancy_forecast.html").is_file(), "docs/occupancy_forecast.html is missing"
    for m in MODULES:
        page = DOCS / "occupancy_forecast" / f"{m}.html"
        assert page.is_file(), f"docs/occupancy_forecast/{m}.html is missing; regenerate the docs (see this file's docstring)"
        html = page.read_text(encoding="utf-8")
        for name, _ in _public_functions_and_classes(_import(m)):
            assert f">{name}<" in html, f"docs for {m} do not mention '{name}'; regenerate the docs"


def _pinned(package):
    lock = ROOT / "requirements-lock.txt"
    if not lock.exists():
        return None
    norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
    for line in lock.read_text().splitlines():
        if "==" in line and not line.startswith("#") and norm(line.split("==")[0].strip()) == norm(package):
            return line.split("==")[1].strip()
    return None


# The HTML depends on pdoc and on the packages it uses to render (for example Pygments writes quotation marks
# differently from one version to the next), so exact comparison is only meaningful when all of them match the lock file.
HTML_TOOLCHAIN = ("pdoc", "pygments", "jinja2", "markupsafe", "markdown2")


def _on_pinned_toolchain():
    try:
        from importlib.metadata import version
        return sys.version_info[:2] == (3, 12) and all(_pinned(p) == version(p) for p in HTML_TOOLCHAIN)
    except Exception:
        return False


@pytest.mark.skipif(not DOCS.exists(), reason="docs/ has not been generated")
@pytest.mark.skipif(not _on_pinned_toolchain(),
                    reason="exact HTML comparison is only enforced on Python 3.12 with the documentation packages pinned in requirements-lock.txt")
def test_committed_html_is_identical_to_a_fresh_regeneration(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "pdoc", "occupancy_forecast", "-o", str(tmp_path), "--docformat", "google"],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    fresh = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    assert fresh, "pdoc produced no files"
    for rel in fresh:
        committed = DOCS / rel
        assert committed.is_file(), f"docs/{rel} is missing; regenerate the docs (see this file's docstring)"
        assert committed.read_bytes() == (tmp_path / rel).read_bytes(), f"docs/{rel} is out of date; regenerate the docs"
