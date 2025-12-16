import click


@click.group("roger")
@click.version_option()
def cli():
    """roger command-line tools"""
    pass
