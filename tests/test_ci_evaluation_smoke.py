"""The offline smoke phase must restore network policy even when a check fails."""

import socket

import pytest
from pytest_socket import SocketConnectBlockedError

from scripts import run_evaluation_smoke


def test_smoke_blocks_remote_connect_and_restores_the_previous_policy(
    tmp_path, monkeypatch
):
    original_connect = socket.socket.connect

    async def try_external_connection(_output_dir):
        with socket.socket() as connection:
            # Documentation-only address: the guard must reject before connecting.
            connection.connect(("203.0.113.1", 443))

    monkeypatch.setattr(run_evaluation_smoke, "_run_checks", try_external_connection)
    with pytest.raises(SocketConnectBlockedError):
        run_evaluation_smoke.run_smoke(tmp_path / "smoke")
    assert socket.socket.connect is original_connect
