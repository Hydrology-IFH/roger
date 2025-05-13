import os
import subprocess
import sys
import filecmp
import fnmatch
from textwrap import dedent


from click.testing import CliRunner
import pytest

import roger.cli

MODELS = (
    "svat",
)


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.mark.parametrize("model", MODELS)
def test_roger_copy_model(model, runner, tmpdir):
    import roger

    result = runner.invoke(roger.cli.roger_copy_model.cli, [model, "--to", os.path.join(tmpdir, model)])
    print(result.output)
    assert result.exit_code == 0, model
    assert not result.output

    outpath = os.path.join(tmpdir, model)
    srcpath = os.path.join(os.path.dirname(roger.__file__), "models", model)
    ignore = [
        f
        for f in os.listdir(srcpath)
        if any(fnmatch.fnmatch(f, pattern) for pattern in roger.cli.roger_copy_model.IGNORE_PATTERNS)
    ]

    comparer = filecmp.dircmp(outpath, srcpath, ignore=ignore)
    assert not comparer.left_only and not comparer.right_only

    with open(os.path.join(outpath, f"{model}.py"), "r") as f:
        model_content = f.read()

    assert "ROGER_VERSION" in model_content

def test_import_isolation(tmpdir):
    TEST_KERNEL = dedent(
        """
    import sys
    import roger.cli

    for mod in sys.modules:
        print(mod)
    """
    )

    tmpfile = tmpdir / "isolation.py"
    with open(tmpfile, "w") as f:
        f.write(TEST_KERNEL)

    proc = subprocess.run([sys.executable, tmpfile], check=True, capture_output=True, text=True)

    imported_modules = proc.stdout.split()
    roger_modules = [mod for mod in imported_modules if mod.startswith("roger.")]

    for mod in roger_modules:
        assert mod.startswith("roger.cli") or mod == "roger._version"

    # make sure using the CLI does not initialize MPI
    assert "mpi4py" not in imported_modules