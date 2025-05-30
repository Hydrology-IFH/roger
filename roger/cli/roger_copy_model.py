#!/usr/bin/env python

from pathlib import Path
import os
import shutil
import datetime
import functools
import textwrap

import click

BASE_PATH = Path(__file__).parent
MODELDIR = BASE_PATH.parent / "models"
MODELDIR_ENVVAR = "ROGER_MODEL_DIR"
IGNORE_PATTERNS = ["__init__.py", "*.pyc", "__pycache__", ".DS_Store", "dummy"]
MODELS = {}

model_dirs = []

for model_dir in os.listdir(MODELDIR):
    path_dir = MODELDIR / model_dir
    if os.path.isdir(path_dir) and model_dir not in IGNORE_PATTERNS:
        model_dirs.append(model_dir)

# populate {model_name: path} mapping
for model_dir in model_dirs:
    MODELS[model_dir] = MODELDIR / model_dir

MODEL_NAMES = sorted(MODELS.keys())


def rewrite_main_file(target_file, model_name):
    from roger import __version__ as roger_version

    try:
        current_date = datetime.datetime.now(datetime.UTC)
    except AttributeError:
        # Fallback for Python < 3.10
        current_date = datetime.datetime.utcnow()
    header_str = textwrap.dedent(
        f'''
        """
        This roger setup file was generated by

           $ roger copy-model {model_name}

        on {current_date:%Y-%m-%d %H:%M:%S} UTC.
        """

        __ROGER_VERSION__ = {roger_version!r}

        if __name__ == "__main__":
            raise RuntimeError(
                "Roger setups cannot be executed directly. "
                f"Try `roger run {{__file__}}` instead."
            )

        # -- end of auto-generated header, original file below --
    '''
    ).strip()

    with open(target_file, "r") as f:
        orig_contents = f.readlines()

    shebang = None
    if orig_contents[0].startswith("#!"):
        shebang = orig_contents[0]
        orig_contents = orig_contents[1:]

    with open(target_file, "w") as f:
        if shebang is not None:
            f.write(shebang + "\n")

        f.write(header_str + "\n\n")
        f.writelines(orig_contents)


def copy_model(model, to=None):
    """Copy a standard setup to another directory.

    Available models:

        {models}

    Example:

        $ roger copy-model svat--to ~/roger-models/svat

    Further directories containing model templates can be added to this command
    via the {model_envvar} environment variable.
    """
    if to is None:
        to = os.path.join(os.getcwd(), model)

    if os.path.exists(to):
        raise RuntimeError("Target directory must not exist")

    to_parent = os.path.dirname(os.path.realpath(to))

    if not os.path.exists(to_parent):
        os.makedirs(to_parent)

    ignore = shutil.ignore_patterns(*IGNORE_PATTERNS)
    shutil.copytree(MODELS[model], to, ignore=ignore)

    main_model_file = os.path.join(to, f"{model}.py")
    rewrite_main_file(main_model_file, model)


copy_model.__doc__ = copy_model.__doc__.format(models=", ".join(MODEL_NAMES), model_envvar=MODELDIR_ENVVAR)


@click.command("roger-copy-model")
@click.argument("model", type=click.Choice(MODEL_NAMES), metavar="MODEL")
@click.option(
    "--to",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=False, writable=True),
    help=("Target directory, must not exist " "(default: copy to current working directory)"),
)
@functools.wraps(copy_model)
def cli(*args, **kwargs):
    copy_model(*args, **kwargs)
