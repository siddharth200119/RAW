import sys
import json
import logging
from unittest.mock import patch
import pytest
from RAW.utils.logger import Logger


def test_logger_json_format(capsys):
    # Mock sys.stdout.isatty to return False to force JSON formatter
    with patch.object(sys.stdout, "isatty", return_value=False):
        logger = Logger("test_json_service", level=logging.INFO)
        logger.info("Hello JSON")

        captured = capsys.readouterr()
        log_lines = captured.out.strip().split("\n")
        assert len(log_lines) >= 1

        # Parse the JSON log line
        log_data = json.loads(log_lines[-1])
        assert log_data["level"] == "INFO"
        assert log_data["service_name"] == "test_json_service"
        assert log_data["message"] == "Hello JSON"
        assert "timestamp" in log_data


def test_logger_colored_format(capsys):
    # Mock sys.stdout.isatty to return True to force colored formatter
    with patch.object(sys.stdout, "isatty", return_value=True):
        logger = Logger("test_colored_service", level=logging.INFO)
        logger.info("Hello Color")

        captured = capsys.readouterr()
        log_output = captured.out
        assert "Hello Color" in log_output
        assert "INFO" in log_output
        assert "test_colored_service" in log_output
        # Reset ANSI code should be present in colored logs
        assert Logger.COLORS["RESET"] in log_output


def test_logger_tracker_id(capsys):
    # Verify tracker_id contextvar is correctly added to log outputs
    with patch.object(sys.stdout, "isatty", return_value=False):
        logger = Logger("test_tracker_service", level=logging.INFO)

        Logger.set_tracker_id("tracker-12345")
        logger.info("Hello Tracker")

        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip().split("\n")[-1])
        assert log_data["tracker_id"] == "tracker-12345"

        # Clear tracker id and verify it is absent
        Logger.set_tracker_id(None)
        logger.info("No Tracker")
        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip().split("\n")[-1])
        assert log_data["tracker_id"] is None


def test_logger_levels(capsys):
    # Verify all logging levels produce correct outputs
    with patch.object(sys.stdout, "isatty", return_value=False):
        logger = Logger("test_levels", level=logging.DEBUG)

        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert len(lines) >= len(levels)

        # Map levels to messages
        for i, level in enumerate(levels):
            log_data = json.loads(lines[i])
            assert log_data["level"] == level
            assert log_data["message"] == f"{level.lower()} msg"
