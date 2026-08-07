"""The suite-wide network guard (tests/conftest.py `_block_network`).

These tests prove the guard is active: any attempt to open a non-local
connection during the suite fails loudly, so a future test that reaches the
network (a real Anthropic call, say) cannot pass silently.
"""

import socket

import pytest


def test_outbound_connection_is_blocked():
    # A documentation IP; the guard raises inside connect before any packet
    # leaves, so this never actually touches the network.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            s.connect(("192.0.2.1", 443))
    finally:
        s.close()
    assert "network access is disabled during tests" in str(excinfo.value)


def test_create_connection_is_blocked():
    # httpx and the Anthropic SDK reach the network through
    # socket.create_connection, which funnels through socket.socket.connect.
    with pytest.raises(RuntimeError) as excinfo:
        socket.create_connection(("192.0.2.1", 443), timeout=1)
    assert "network access is disabled during tests" in str(excinfo.value)


def test_connect_ex_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError):
            s.connect_ex(("192.0.2.1", 443))
    finally:
        s.close()


def test_dns_resolution_is_blocked():
    # The guard blocks getaddrinfo for non-local hosts too: without it, a
    # test reaching for a real hostname would perform a real DNS lookup
    # before the connect guard fires, leaking the hostname to the resolver.
    with pytest.raises(RuntimeError, match="DNS resolution"):
        socket.getaddrinfo("api.anthropic.com", 443)


def test_gethostbyname_is_blocked():
    # gethostbyname resolves a name without going through getaddrinfo, so it
    # is guarded on its own to close the same DNS-leak gap.
    with pytest.raises(RuntimeError, match="DNS resolution"):
        socket.gethostbyname("api.anthropic.com")


def test_gethostbyname_ex_is_blocked():
    with pytest.raises(RuntimeError, match="DNS resolution"):
        socket.gethostbyname_ex("api.anthropic.com")


def test_udp_sendto_is_blocked():
    # UDP is connectionless: sendto carries the destination directly and never
    # touches connect, so it must be guarded on its own or it bypasses the
    # connect guard.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            s.sendto(b"x", ("192.0.2.1", 443))
    finally:
        s.close()
    assert "network access is disabled during tests" in str(excinfo.value)


def test_local_dns_resolution_allowed():
    assert socket.getaddrinfo("localhost", 0)


def test_local_gethostbyname_allowed():
    assert socket.gethostbyname("localhost")
