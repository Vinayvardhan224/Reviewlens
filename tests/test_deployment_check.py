"""
Unit tests for ReviewLens Deployment Verification Script (scripts/check_deployment.py).
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_deployment import (
    check_pwa_config,
    check_manifest,
    check_requirements_consistency,
    check_git_untracked_secrets,
    check_for_leaked_secrets,
    main
)


class TestDeploymentCheck(unittest.TestCase):

    def test_pwa_config_passes(self):
        self.assertTrue(check_pwa_config())

    def test_manifest_passes(self):
        self.assertTrue(check_manifest())

    def test_requirements_consistency_passes(self):
        self.assertTrue(check_requirements_consistency())

    def test_git_untracked_secrets_passes(self):
        self.assertTrue(check_git_untracked_secrets())

    def test_no_leaked_secrets_passes(self):
        self.assertTrue(check_for_leaked_secrets())

    def test_main_exits_zero(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
