import json, pytest
from ragv1.evaluation.qaset import load_qaset, QASetError


def _write(tmp_path, rows):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_loads_valid_rows(tmp_path):
    p = _write(tmp_path, [{"question": "怎么配？", "answer_chunk_ids": ["a", "b"], "category": "multi_hop"}])
    items = load_qaset(p)
    assert len(items) == 1 and items[0].answer_chunk_ids == ("a", "b")


def test_row_without_history_gets_empty_tuple(tmp_path):
    p = _write(tmp_path, [{"question": "q", "answer_chunk_ids": ["a"], "category": "fact"}])
    assert load_qaset(p)[0].history == ()


def test_unanswerable_row_may_have_empty_answer(tmp_path):
    p = _write(tmp_path, [{"question": "库里没有的", "answer_chunk_ids": [], "category": "unanswerable"}])
    assert load_qaset(p)[0].answer_chunk_ids == ()


def test_rejects_missing_question(tmp_path):
    p = _write(tmp_path, [{"answer_chunk_ids": ["a"], "category": "fact"}])
    with pytest.raises(QASetError, match="question"):
        load_qaset(p)


def test_rejects_blank_line_noise(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"question":"q","answer_chunk_ids":["a"],"category":"fact"}\n\n', encoding="utf-8")
    assert len(load_qaset(p)) == 1


def test_reports_line_number_on_bad_json(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"question":"q","answer_chunk_ids":["a"],"category":"fact"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(QASetError, match="第 2 行"):
        load_qaset(p)
