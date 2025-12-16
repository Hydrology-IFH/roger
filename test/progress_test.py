import sys
import time
import platform
import pytest


class Dummy:
    pass


@pytest.mark.xfail(platform.system() == "Darwin", reason="Flaky on OSX")
def test_progress_format(capsys):
    from roger.logs import setup_logging

    setup_logging(stream_sink=sys.stdout)

    from roger.progress import get_progress_bar

    dummy_state = Dummy()
    dummy_state.settings = Dummy()
    dummy_state.variables = Dummy()
    dummy_state.settings.warmup_done = True
    dummy_state.settings.runlen = 8000
    dummy_state.variables.time = 0
    dummy_state.variables.itt = 0

    with get_progress_bar(dummy_state, use_tqdm=False) as pbar:
        for _ in range(8):
            time.sleep(0.1)
            pbar.advance_time(1000)

    captured_log = capsys.readouterr()
    assert "Current iteration:" in captured_log.out

    with get_progress_bar(dummy_state, use_tqdm=True) as pbar:
        for _ in range(8):
            time.sleep(0.1)
            pbar.advance_time(1000)

    captured_tqdm = capsys.readouterr()
    assert "Current iteration:" in captured_tqdm.out