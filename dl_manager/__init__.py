from .cli import main as _main

def run_cli_app(port=9011, script='', invalidate_checkpoints=False):
    _main(port, script, invalidate_checkpoints)

