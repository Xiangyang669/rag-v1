"""多模态通道：结构化描述 + 宽松的 JSON 解析。

VLM 的可靠性只体现在一件事上：**它经常不按格式返回**。可能裹 ```` ```json ````
围栏、可能在 JSON 前后写一段客套话。解析必须宽松，但**不能丢内容**——
解析不出来时把原始返回当描述保留，那也比丢掉整张图强。
"""

from ragv1.ingest.vlm import build_vision_engine, parse_vision_json

GOOD = '{"image_type":"流程图","content":"三步流程","data":"","relations":"A→B→C"}'


def test_parse_plain_json():
    r = parse_vision_json(GOOD)
    assert r.image_type == "流程图" and r.content == "三步流程" and r.relations == "A→B→C"


def test_parse_strips_markdown_json_fence():
    """VLM 爱把 JSON 裹进 ```json 围栏"""
    r = parse_vision_json(f"```json\n{GOOD}\n```")
    assert r is not None and r.content == "三步流程"


def test_parse_extracts_json_from_surrounding_prose():
    r = parse_vision_json(f"好的，结果如下：\n{GOOD}\n希望有帮助。")
    assert r is not None and r.image_type == "流程图"


def test_parse_returns_none_on_garbage():
    assert parse_vision_json("这不是 JSON") is None


def test_parse_missing_keys_defaults_to_empty():
    r = parse_vision_json('{"image_type":"图表"}')
    assert r is not None and r.content == "" and r.data == ""


def test_build_returns_none_without_api_key(monkeypatch):
    """没有凭据时干净降级为 None，不抛异常也不给半成品引擎。

    只 delenv 不够：凭据还可能来自 .env 文件——本机若配了 key，这条会假失败。
    把候选文件列表也清空，才真的构造出「完全没有凭据」这个条件。
    """
    import ragv1.embedding as embedding

    monkeypatch.delenv(embedding.KEY_NAME, raising=False)
    monkeypatch.setattr(embedding, "ENV_FILE_CANDIDATES", ())
    assert build_vision_engine("siliconflow") is None


def test_build_returns_none_for_unknown_engine():
    assert build_vision_engine("no-such-engine") is None


def test_describe_sends_data_url_and_parses_reply():
    from ragv1.ingest.vlm import SiliconFlowVision

    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)

            class Msg:
                content = GOOD

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    eng = SiliconFlowVision(client=FakeClient())
    r = eng.describe(b"\x89PNG-fake")
    assert r.content == "三步流程"
    url = captured["messages"][0]["content"][1]["image_url"]["url"]
    assert "data:image/png;base64," in url
