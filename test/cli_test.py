import os
import filecmp
import fnmatch
__requires__ = ["CherryPy < 3"]
import pkg_resources

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
    result = runner.invoke(roger.cli.roger_copy_model.cli, [model, "--to", os.path.join(tmpdir, model)])
    print(result.output)
    assert result.exit_code == 0, model
    assert not result.output

    outpath = os.path.join(tmpdir, model)
    srcpath = pkg_resources.resource_filename("roger", f"models/{model}")
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
