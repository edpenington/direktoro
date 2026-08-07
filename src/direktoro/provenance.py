"""Provenance of the running code itself: which copy of direktoro is loaded.

`direktoro.routing` records the provenance of a CALL — where it went and what
came back. This module records the provenance of the caller's engine: one
digest over the package's own source files.

Pure standard library, no network, and it reads nothing outside the imported
package's directory.
"""

import hashlib
from pathlib import Path


def _hash_tree(directory):
    """One sha256 over every `*.py` under `directory`, or None if there is none.

    Files are visited in sorted relative-POSIX-path order, and each contributes
    `relpath\\x00bytes`, so the digest depends on the file names and their
    contents and on nothing else — not on filesystem walk order, not on where
    the directory happens to sit. `__pycache__` directories and compiled `.pyc`
    files are skipped: they are derived from the source, they differ between
    interpreters, and hashing them would make one checkout report two digests.

    None means there was nothing to hash or it could not be read — a path that
    is not a directory, a directory holding no source, a file that vanished or
    refused to open mid-walk. A digest over zero files would otherwise be a
    fixed constant claiming to identify whatever produced it.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    try:
        paths = sorted(
            (path for path in root.rglob("*.py")
             if "__pycache__" not in path.parts),
            key=lambda path: path.relative_to(root).as_posix())
        if not paths:
            return None
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def source_hash():
    """A sha256 hex digest of the imported package's own source files.

    Identifies the exact source of the running copy, for a consumer that folds
    engine identity into a run's provenance: the version string names a
    release, this names the bytes that produced the run, so an edited or
    patched checkout is distinguishable from the release it started as. Returns
    the token `"nosource"` when the package's source cannot be read, as for a
    frozen or zipimported copy.
    """
    here = globals().get("__file__")
    if here is None:
        return "nosource"
    digest = _hash_tree(Path(here).parent)
    return "nosource" if digest is None else digest
