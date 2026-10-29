import pytest

from RAW import __version__
from RAW.cli import MARKER, SKILL_NAME, main
from RAW.models import Skill, bundled_skills_dir


def run(*argv):
    return main(list(argv))


def test_install_copies_a_loadable_skill_and_marks_it(tmp_path, capsys):
    assert run("install-skill", "--dest", str(tmp_path)) == 0

    installed = tmp_path / SKILL_NAME
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "agent.md").is_file()
    assert (installed / MARKER).read_text().strip() == __version__
    assert not list(tmp_path.glob(".*installing*"))
    assert not list(installed.rglob("__pycache__"))
    skill = Skill.from_directory(installed)
    assert skill.name == "raw-agents"
    assert str(installed) in capsys.readouterr().out


def test_reinstall_refreshes_a_copy_it_created(tmp_path):
    run("install-skill", "--dest", str(tmp_path))
    installed = tmp_path / SKILL_NAME
    (installed / "SKILL.md").write_text("stale")
    (installed / "leftover.md").write_text("old file")

    assert run("install-skill", "--dest", str(tmp_path)) == 0

    assert "raw-agents" in (installed / "SKILL.md").read_text()
    assert not (installed / "leftover.md").exists()


def test_a_folder_it_did_not_create_is_protected_unless_forced(tmp_path, capsys):
    mine = tmp_path / SKILL_NAME
    mine.mkdir()
    (mine / "SKILL.md").write_text("my own skill")

    assert run("install-skill", "--dest", str(tmp_path)) == 1
    assert (mine / "SKILL.md").read_text() == "my own skill"
    assert "--force" in capsys.readouterr().err

    assert run("install-skill", "--dest", str(tmp_path), "--force") == 0
    assert "raw-agents" in (mine / "SKILL.md").read_text()
    assert (mine / MARKER).is_file()


def test_user_scope_uses_home_and_claude_config_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert run("install-skill") == 0
    assert (home / ".claude" / "skills" / SKILL_NAME / "SKILL.md").is_file()

    config = tmp_path / "elsewhere"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    assert run("install-skill") == 0
    assert (config / "skills" / SKILL_NAME / "SKILL.md").is_file()


def test_project_scope_uses_the_current_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    assert run("install-skill", "--scope", "project") == 0

    assert (project / ".claude" / "skills" / SKILL_NAME / "SKILL.md").is_file()


def test_uninstall_removes_only_what_it_installed(tmp_path, capsys):
    assert run("uninstall-skill", "--dest", str(tmp_path)) == 1
    assert "Nothing to remove" in capsys.readouterr().err

    run("install-skill", "--dest", str(tmp_path))
    assert run("uninstall-skill", "--dest", str(tmp_path)) == 0
    assert not (tmp_path / SKILL_NAME).exists()

    foreign = tmp_path / SKILL_NAME
    foreign.mkdir()
    assert run("uninstall-skill", "--dest", str(tmp_path)) == 1
    assert foreign.exists()
    assert run("uninstall-skill", "--dest", str(tmp_path), "--force") == 0
    assert not foreign.exists()


def test_skill_path_prints_the_packaged_folder(capsys):
    assert run("skill-path") == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(bundled_skills_dir() / SKILL_NAME)


def test_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        run("nope")
    assert e.value.code == 2
