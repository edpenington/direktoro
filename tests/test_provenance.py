"""Tests for engine provenance (direktoro.provenance).

`source_hash()` answers "which bytes of direktoro produced this run?", so the
two properties that matter are that it is stable for one copy of the source and
that it moves for another. Both are checked against a temporary tree rather
than the installed package, which cannot be edited mid-test — `_hash_tree` is
the digest itself, and `source_hash` is that digest applied to the package's own
directory.
"""

import re

from direktoro.provenance import _hash_tree, source_hash


def _write_package(root, body="VALUE = 1\n"):
    """A miniature package tree: two modules and a subpackage."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text(body, encoding="utf-8")
    (root / "module.py").write_text("def f():\n    return 42\n",
                                    encoding="utf-8")
    sub = root / "sub"
    sub.mkdir(exist_ok=True)
    (sub / "__init__.py").write_text("SUB = True\n", encoding="utf-8")
    return root


class TestSourceHash:
    def test_it_is_the_same_answer_twice(self):
        assert source_hash() == source_hash()

    def test_it_is_a_bare_sha256_hex_digest(self):
        assert re.fullmatch(r"[0-9a-f]{64}", source_hash())

    def test_unreadable_source_answers_nosource(self, monkeypatch):
        # A frozen or zipimported copy has no directory to walk, so the digest
        # is unavailable and the token says so in the record.
        monkeypatch.setattr("direktoro.provenance._hash_tree",
                            lambda directory: None)
        assert source_hash() == "nosource"


class TestHashTree:
    def test_the_same_tree_hashes_the_same(self, tmp_path):
        first = _write_package(tmp_path / "one")
        second = _write_package(tmp_path / "two")
        # Two identical trees in different directories: the digest covers the
        # relative paths and the bytes, and nothing about where they sit.
        assert _hash_tree(first) == _hash_tree(second)
        assert _hash_tree(first) == _hash_tree(first)

    def test_one_changed_byte_changes_the_digest(self, tmp_path):
        root = _write_package(tmp_path / "pkg")
        before = _hash_tree(root)
        (root / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert _hash_tree(root) != before

    def test_a_renamed_file_changes_the_digest(self, tmp_path):
        # The path is hashed alongside the bytes, so moving code between
        # modules is a different engine even when every byte survives.
        root = _write_package(tmp_path / "pkg")
        before = _hash_tree(root)
        (root / "module.py").rename(root / "renamed.py")
        assert _hash_tree(root) != before

    def test_added_and_removed_files_change_the_digest(self, tmp_path):
        root = _write_package(tmp_path / "pkg")
        before = _hash_tree(root)
        (root / "extra.py").write_text("EXTRA = 0\n", encoding="utf-8")
        with_extra = _hash_tree(root)
        assert with_extra != before
        (root / "extra.py").unlink()
        assert _hash_tree(root) == before

    def test_compiled_droppings_are_not_part_of_the_source(self, tmp_path):
        # `__pycache__` contents are derived and differ between interpreters;
        # hashing them would give one checkout two digests.
        root = _write_package(tmp_path / "pkg")
        before = _hash_tree(root)
        cache = root / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-311.pyc").write_bytes(b"\x00compiled")
        (cache / "shadow.py").write_text("SHADOW = 1\n", encoding="utf-8")
        (root / "notes.txt").write_text("not source\n", encoding="utf-8")
        assert _hash_tree(root) == before

    def test_a_tree_with_no_source_in_it_has_no_digest(self, tmp_path):
        # A digest over zero files is a fixed constant, and a run recording it
        # would look identical to every other run that found nothing.
        assert _hash_tree(tmp_path / "does-not-exist") is None
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _hash_tree(empty) is None
