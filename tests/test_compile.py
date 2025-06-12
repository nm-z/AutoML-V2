import os
import py_compile
import pytest

# Collect all Python files in the repository
python_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(os.path.dirname(os.path.dirname(__file__)))
    for f in files if f.endswith('.py') and 'env-' not in root
]

@pytest.mark.parametrize("path", python_files)
def test_python_file_compiles(path):
    """Ensure all Python files are syntactically valid."""
    py_compile.compile(path, doraise=True)
