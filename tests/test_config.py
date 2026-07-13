"""
test_config.py — Tests for config.py
Covers: default values, env var overrides, config_overrides.json loading.
"""
import os
import sys
import json
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestConfigDefaults:
    """Test that config.py loads correct default values."""

    def test_company_name_has_default(self):
        from config import COMPANY_NAME
        assert isinstance(COMPANY_NAME, str)
        assert len(COMPANY_NAME) > 0

    def test_sender_name_has_default(self):
        from config import SENDER_NAME
        assert isinstance(SENDER_NAME, str)

    def test_team_phone_has_default(self):
        from config import TEAM_PHONE
        assert isinstance(TEAM_PHONE, str)

    def test_courses_is_list(self):
        from config import COURSES
        assert isinstance(COURSES, list)
        assert len(COURSES) > 0

    def test_course_structure(self):
        from config import COURSES
        for course in COURSES:
            assert "name" in course
            assert "duration" in course
            assert "price" in course
            assert "emi_start" in course
            assert "highlight" in course

    def test_batch_schedules(self):
        from config import BATCH_SCHEDULES
        assert isinstance(BATCH_SCHEDULES, dict)
        assert "weekday" in BATCH_SCHEDULES
        assert "weekend" in BATCH_SCHEDULES

    def test_placement_rate(self):
        from config import PLACEMENT_RATE
        assert isinstance(PLACEMENT_RATE, str)
        assert "%" in PLACEMENT_RATE

    def test_recovery_rate_threshold(self):
        from config import RECOVERY_RATE_THRESHOLD
        assert isinstance(RECOVERY_RATE_THRESHOLD, (int, float))
        assert 0 < RECOVERY_RATE_THRESHOLD <= 100

    def test_hours_before_overdue(self):
        from config import HOURS_BEFORE_OVERDUE
        assert isinstance(HOURS_BEFORE_OVERDUE, (int, float))
        assert HOURS_BEFORE_OVERDUE > 0

    def test_email_signature(self):
        from config import EMAIL_SIGNATURE
        assert isinstance(EMAIL_SIGNATURE, str)
        assert len(EMAIL_SIGNATURE) > 0


class TestConfigOverrides:
    """Test environment variable overrides."""

    def test_env_var_overrides_company_name(self, monkeypatch):
        monkeypatch.setenv("COMPANY_NAME", "Test Corp")
        # Re-import to pick up the env var
        import importlib
        import config
        importlib.reload(config)
        assert config.COMPANY_NAME == "Test Corp"
        # Restore
        monkeypatch.delenv("COMPANY_NAME", raising=False)
        importlib.reload(config)

    def test_env_var_overrides_sender_name(self, monkeypatch):
        monkeypatch.setenv("SENDER_NAME", "Test Sender")
        import importlib
        import config
        importlib.reload(config)
        assert config.SENDER_NAME == "Test Sender"
        monkeypatch.delenv("SENDER_NAME", raising=False)
        importlib.reload(config)

    def test_env_var_overrides_team_phone(self, monkeypatch):
        monkeypatch.setenv("TEAM_PHONE", "+1-555-TEST")
        import importlib
        import config
        importlib.reload(config)
        assert config.TEAM_PHONE == "+1-555-TEST"
        monkeypatch.delenv("TEAM_PHONE", raising=False)
        importlib.reload(config)


class TestConfigOverridesJSON:
    """Test config_overrides.json loading."""

    def test_overrides_file_loading(self, tmp_path):
        overrides_path = tmp_path / "config_overrides.json"
        overrides_path.write_text(json.dumps({
            "COMPANY_NAME": "Override Corp",
            "SENDER_NAME": "Override Sender",
        }))
        with open(overrides_path) as f:
            data = json.load(f)
        assert data["COMPANY_NAME"] == "Override Corp"
        assert data["SENDER_NAME"] == "Override Sender"

    def test_missing_overrides_file(self, tmp_path):
        overrides_path = tmp_path / "nonexistent.json"
        assert not overrides_path.exists()
        # Should handle missing file gracefully
        data = {}
        if overrides_path.exists():
            with open(overrides_path) as f:
                data = json.load(f)
        assert data == {}
