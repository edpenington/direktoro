"""Shared test configuration.

This module installs a network guard (see `_block_network` below) so the
suite's zero-network property is enforced mechanically, not just left as a
consequence of lazy client construction. direktoro makes no LLM call in its
tests: every adapter is exercised against stubbed clients and hand-authored
request/response fixtures, and `build_adapter` reads an injected env, never the
process environment or the network.
"""

import socket

import pytest


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------
# The suite must never touch the network or need an API key: every LLM call
# is stubbed. This guard makes that property fail loudly instead of relying
# on it as an accident of the code path. We patch the connect primitives on
# `socket.socket` (which `socket.create_connection`, httpx, and the Anthropic
# SDK all funnel through), the name-resolution functions (`socket.getaddrinfo`,
# `socket.gethostbyname`, `socket.gethostbyname_ex`), and the connectionless
# UDP `socket.socket.sendto` path, so every route to a non-local destination
# raises. Loopback and AF_UNIX addresses are allowed so anything genuinely
# local still works, though nothing in the suite needs even that today.
#
# A hand-rolled guard is preferred over the pytest-socket dependency: it is a
# dozen lines, keeps the test dependencies at zero third-party packages, and
# raises a message that points straight at this file. pytest-socket would add
# an install for no capability this does not already cover.

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname
_real_gethostbyname_ex = socket.gethostbyname_ex


def _guarded_resolver(resolver):
    # Block name resolution of non-local hosts: without this, a test reaching
    # for a real hostname performs a real DNS lookup before the connect guard
    # fires, leaking the hostname to the resolver. getaddrinfo, gethostbyname,
    # and gethostbyname_ex all take the host as their first argument.
    def _guard(host, *args, **kwargs):
        if host is None or (isinstance(host, (str, bytes)) and (
                host.decode() if isinstance(host, bytes) else host)
                in _ALLOWED_HOSTS):
            return resolver(host, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted DNS "
            f"resolution of {host!r}). The suite must never touch the network "
            "or need an API key: stub the client instead. See the network "
            "guard in tests/conftest.py."
        )
    return _guard


def _is_local_address(address):
    # AF_UNIX addresses are str/bytes filesystem paths: always local.
    if isinstance(address, (str, bytes)):
        return True
    # AF_INET / AF_INET6 addresses are (host, port[, ...]) tuples.
    if isinstance(address, tuple) and address:
        return address[0] in _ALLOWED_HOSTS
    return False


def _blocked(operation):
    def _guard(self, address, *args, **kwargs):
        if _is_local_address(address):
            return operation(self, address, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted connection "
            f"to {address!r}). The suite must never touch the network or need "
            "an API key: stub the client instead. See the network guard in "
            "tests/conftest.py."
        )
    return _guard


def _blocked_sendto(operation):
    # UDP is connectionless: sendto carries the destination directly and never
    # touches connect, so it would bypass the connect guard. sendto(data,
    # address) and sendto(data, flags, address) both put the address last
    # positionally.
    def _guard(self, *args, **kwargs):
        address = args[-1] if args else None
        if address is None or _is_local_address(address):
            return operation(self, *args, **kwargs)
        raise RuntimeError(
            "network access is disabled during tests (attempted datagram send "
            f"to {address!r}). The suite must never touch the network or need "
            "an API key: stub the client instead. See the network guard in "
            "tests/conftest.py."
        )
    return _guard


@pytest.fixture(scope="session", autouse=True)
def _block_network():
    """Fail any test that opens a non-local network connection.

    Session-scoped and autouse so it wraps every test without opt-in. The
    patched primitives are restored afterwards so the guard does not leak
    out of the pytest process.
    """
    socket.socket.connect = _blocked(_real_connect)
    socket.socket.connect_ex = _blocked(_real_connect_ex)
    socket.socket.sendto = _blocked_sendto(_real_sendto)
    socket.getaddrinfo = _guarded_resolver(_real_getaddrinfo)
    socket.gethostbyname = _guarded_resolver(_real_gethostbyname)
    socket.gethostbyname_ex = _guarded_resolver(_real_gethostbyname_ex)
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex
        socket.socket.sendto = _real_sendto
        socket.getaddrinfo = _real_getaddrinfo
        socket.gethostbyname = _real_gethostbyname
        socket.gethostbyname_ex = _real_gethostbyname_ex
