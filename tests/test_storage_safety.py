"""
Unit tests for ReviewLens Account Storage Safety guarantees.
Verifies thread-safety, atomic file replacement with tempfile + os.replace,
and concurrent write protection.
"""

import unittest
import threading
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from modules.auth import _save_user_db, _load_user_db, USER_DB_PATH, _USER_DB_LOCK


class TestStorageSafety(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.mock_db_path = Path(self.test_dir.name) / "users.json"
        self.patcher = patch("modules.auth.USER_DB_PATH", self.mock_db_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.test_dir.cleanup()

    def test_atomic_save_and_load(self):
        """Verifies that user data is atomically saved and successfully read back."""
        data = {
            "test_user": {
                "email": "test@reviewlens.com",
                "tier": "Power Reviewer",
                "salt": "abc123salt",
                "pw_hash": "deadbeef"
            }
        }
        _save_user_db(data)
        self.assertTrue(self.mock_db_path.exists())

        loaded = _load_user_db()
        self.assertEqual(loaded, data)
        self.assertIn("test_user", loaded)

    def test_concurrent_writes_do_not_corrupt_database(self):
        """Simulates concurrent threads writing user accounts under lock."""
        num_threads = 5
        errors = []

        def worker(thread_idx: int):
            try:
                for i in range(2):
                    with _USER_DB_LOCK:
                        db = _load_user_db()
                        db[f"user_{thread_idx}_{i}"] = {
                            "email": f"user_{thread_idx}_{i}@test.com",
                            "created": i
                        }
                        _save_user_db(db)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive(), "Worker thread timed out / deadlocked!")

        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
        final_db = _load_user_db()
        self.assertEqual(len(final_db), num_threads * 2)


if __name__ == "__main__":
    unittest.main()
