"""Run with: pytest -q   (no model or network needed)"""
import os

os.environ["AGENT_DB"] = ":memory:"
os.environ["AGENT_LOG"] = os.devnull

import pytest

import agent


def test_calc_basic():
    assert agent.calc("2+3*4") == "14"
    assert agent.calc("2**10") == "1024"


def test_calc_blocks_code_execution():
    with pytest.raises(ValueError):
        agent.calc("__import__('os').system('echo hi')")


def test_calc_blocks_huge_power():
    with pytest.raises(ValueError):
        agent.calc("9**999")


def test_read_file_blocks_path_escape():
    assert agent.read_file("../agent.py") == "Access denied."


def test_read_file_ok():
    p = agent.FILES_DIR / "_test_file.txt"
    p.write_text("hello")
    try:
        assert agent.read_file("_test_file.txt") == "hello"
    finally:
        p.unlink()


def test_notes_roundtrip():
    agent.save_note("favourite club: barcelona")
    assert "barcelona" in agent.recall_notes("club")


def test_history_roundtrip_and_clear():
    agent.save_turn("t1", "hi", "hello")
    assert agent.load_history("t1")[-1] == {"role": "assistant", "content": "hello"}
    agent.clear_history("t1")
    assert agent.load_history("t1") == []


def test_history_is_capped():
    for i in range(40):
        agent.save_turn("t2", f"q{i}", f"a{i}")
    count = agent._db("SELECT COUNT(*) FROM messages WHERE sid='t2'")[0][0]
    assert count == agent.HISTORY_STORE_LIMIT
