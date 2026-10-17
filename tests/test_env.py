import os

from cia.env import load_env, parse


def test_parse_skips_comments_and_strips_quotes_and_export():
    text = "# a comment\n\nA=1\nexport B=\"two words\"\nC='x # not a comment'\nD=3  # inline\nnot a line\nE=\n"
    assert parse(text) == {
        "A": "1",
        "B": "two words",
        "C": "x # not a comment",
        "D": "3",
        "E": "",
    }


def test_load_env_never_overrides_the_shell(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("CIA_T_SHELL=from-file\nCIA_T_NEW=from-file\n")
    monkeypatch.setenv("CIA_T_SHELL", "from-shell")
    monkeypatch.delenv("CIA_T_NEW", raising=False)
    assert load_env(f) == ["CIA_T_NEW"]
    assert (
        os.environ["CIA_T_SHELL"] == "from-shell"
        and os.environ["CIA_T_NEW"] == "from-file"
    )
    monkeypatch.delenv("CIA_T_NEW")


def test_a_missing_file_is_fine(tmp_path):
    assert load_env(tmp_path / "nope") == []


def test_the_example_lists_every_setting_the_code_reads():
    from pathlib import Path

    from cia import llm

    keys = parse(Path(".env.example").read_text().replace("# CIA_", "CIA_"))
    for name in (
        llm.OPENAI_BASE_URL_ENV,
        llm.OPENAI_MODEL_ENV,
        llm.OPENAI_API_KEY_ENV,
        llm.OPENAI_MAX_TOKENS_ENV,
        llm.OPENROUTER_API_KEY_ENV,
        llm.OPENROUTER_MODEL_ENV,
        llm.MODEL_ID_ENV,
    ):
        assert name in keys, name
