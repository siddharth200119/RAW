"""
``raw-agents`` command line: installs the skill that ships with the package.

The skill teaches an AI coding assistant (Claude Code and other tools that read the
``SKILL.md`` format) how to use this library. Run ``raw-agents install-skill`` after
``pip install raw-agents``, and again after upgrading the package to refresh it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from RAW import __version__
from RAW.models.skill import SKILL_FILE, bundled_skills_dir

SKILL_NAME = "raw-agents"
# Written into installed copies; its presence means raw-agents installed the folder and may replace it.
MARKER = ".raw-agents-installed"


class CliError(Exception):
    pass


def _say(message: str = "", error: bool = False) -> None:
    (sys.stderr if error else sys.stdout).write(message + "\n")


def _claude_skills_dir(scope: str) -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "skills"
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(config) if config else Path.home() / ".claude") / "skills"


def _destination(scope: str, dest: Optional[str]) -> Path:
    return Path(dest).expanduser() if dest else _claude_skills_dir(scope)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def install_skill(skills_dir: Path, force: bool = False) -> Path:
    """Copy the bundled skill to ``skills_dir/raw-agents`` and return that path."""
    source = bundled_skills_dir() / SKILL_NAME
    if not (source / SKILL_FILE).is_file():
        raise CliError(f"The package does not contain the skill at {source}; reinstall raw-agents.")
    target = skills_dir / SKILL_NAME
    if (target.exists() or target.is_symlink()) and not (target / MARKER).is_file() and not force:
        raise CliError(f"{target} already exists and was not installed by raw-agents. Use --force to replace it.")

    skills_dir.mkdir(parents=True, exist_ok=True)
    staging = skills_dir / f".{SKILL_NAME}.installing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (staging / MARKER).write_text(__version__ + "\n", encoding="utf-8")
        if target.exists() or target.is_symlink():
            _remove(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def uninstall_skill(skills_dir: Path, force: bool = False) -> Path:
    target = skills_dir / SKILL_NAME
    if not (target.exists() or target.is_symlink()):
        raise CliError(f"Nothing to remove: {target} does not exist.")
    if not (target / MARKER).is_file() and not force:
        raise CliError(f"{target} was not installed by raw-agents. Use --force to remove it anyway.")
    _remove(target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raw-agents", description="Tools for the raw-agents package.")
    parser.add_argument("--version", action="version", version=f"raw-agents {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def location(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--scope", choices=["user", "project"], default="user",
            help="user: ~/.claude/skills (or $CLAUDE_CONFIG_DIR/skills), for every project. "
                 "project: ./.claude/skills, for the current project only. Default: user.",
        )
        sub.add_argument("--dest", help="Install into this skills directory instead (for other tools).")
        sub.add_argument("--force", action="store_true", help="Replace a folder raw-agents did not create.")

    location(commands.add_parser("install-skill", help="Install the raw-agents skill for Claude Code."))
    location(commands.add_parser("uninstall-skill", help="Remove the installed raw-agents skill."))
    commands.add_parser("skill-path", help="Print the folder of the skill inside the installed package.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "skill-path":
            _say(str(bundled_skills_dir() / SKILL_NAME))
        elif args.command == "install-skill":
            target = install_skill(_destination(args.scope, args.dest), args.force)
            _say(f"Installed the raw-agents skill (version {__version__}) to {target}")
            _say("Start a new Claude Code session to pick it up. Re-run this command after upgrading raw-agents.")
        else:
            _say(f"Removed {uninstall_skill(_destination(args.scope, args.dest), args.force)}")
    except CliError as e:
        _say(f"error: {e}", error=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
