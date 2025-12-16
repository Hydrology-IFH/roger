#!/usr/bin/env python

try:
    import click

    have_click = True
except ImportError:
    have_click = False

if not have_click:
    raise ImportError("The Roger command line tools require click (e.g. through `pip install click`)")

del click
del have_click

from roger.cli import roger, roger_run, roger_copy_model, roger_create_mask, roger_resubmit  # noqa: E402

roger.cli.add_command(roger_run.cli, "run")
roger.cli.add_command(roger_copy_model.cli, "copy-model")
roger.cli.add_command(roger_create_mask.cli, "create-mask")
roger.cli.add_command(roger_resubmit.cli, "resubmit")
