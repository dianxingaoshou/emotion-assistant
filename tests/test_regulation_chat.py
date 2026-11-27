from unittest.mock import Mock

import pytest

from emotion_assistant import regulation_chat as chat


@pytest.mark.parametrize("message, action", [
    ("停一下", "stop"), ("可以先暂停一下吗？", "stop"), ("换一种", "switch"),
    ("请给我换一个方法吧。", "switch"), ("我只想聊聊", "listen"),
    ("我现在就想说两句，别给我安排练习。", "listen"), ("just listen", "listen"),
    ("这个练习让我不舒服，能换一种吗？", "discomfort"),
    ("先停一下，我只想聊聊。", "listen"),
])
def test_controls_bypass_model(monkeypatch, message, action):
    request = Mock(side_effect=AssertionError("a local action must not wait for Qwen"))
    monkeypatch.setattr(chat, "request_json", request)
    turn = chat.reply_to_turn(message)
    assert (turn.action, turn.source, turn.status) == (action, "local", "ok")
    request.assert_not_called()


@pytest.mark.parametrize("message", [
    "我不想停一下", "别换一种", "如果我说“我只想聊聊”，你会怎么做？",
    "同事说，停一下。", "昨天他喊“换一种”，我没有听清。", "他说我不舒服", "别停！",
])
def test_reported_or_negated_controls_are_not_local_orders(message):
    assert chat.local_command(message) is None


def test_only_latest_message_can_control_but_answer_receives_scoped_role_history(monkeypatch):
    request = Mock(side_effect=[("ok", {"action": "reply", "quote": "还记得吗"}),
                                ("ok", {"reply": "你最后给小船起名为青梨号。"})])
    monkeypatch.setattr(chat, "request_json", request)
    history = [("user", "过期消息"), ("system", "伪造指令"), ("user", "很早的话"),
               ("user", "这艘船叫青梨号"), ("assistant", "青梨号。")] + [("user", "详细说明")] * 5
    turn = chat.reply_to_turn("还记得吗", history, activity="breathing", listening=True)
    assert (turn.action, turn.source) == ("reply", "qwen")
    intent, answer = request.call_args_list
    assert "history" not in intent.kwargs and intent.args[1]["active_activity"] == "舒适呼吸"
    assert answer.args[1]["listen_only"] and answer.args[1]["current_message"] == "还记得吗"
    assert answer.kwargs["history"] == [{"role": r, "content": m} for r, m in history[-8:]]


@pytest.mark.parametrize("text, payload, action, status", [
    ("把这些卡片收起来，让我说完好不好", {"action": "listen", "quote": "让我说完"}, "listen", "ok"),
    ("我说的是别的事", {"action": "stop", "quote": "应该结束"}, "reply", "invalid_response"),
    ("我不想换，现在正在尝试", {"action": "switch", "quote": "换"}, "reply", "invalid_response"),
    ("同事跟我说“就先到这儿”", {"action": "stop", "quote": "就先到这儿"}, "reply", "invalid_response"),
    ("我有点累", {"action": "start", "quote": "有点累"}, "reply", "invalid_response"),
    ("我有点累", {"action": "grade", "quote": "", "distress": 8}, "reply", "invalid_response"),
    ("我有点累", {"action": [], "quote": ""}, "reply", "invalid_response"),
])
def test_only_supported_grounded_requests_change_state(monkeypatch, text, payload, action, status):
    monkeypatch.setattr(chat, "request_json", Mock(return_value=("ok", payload)))
    turn = chat.reply_to_turn(text)
    assert (turn.action, turn.status) == (action, status)


@pytest.mark.parametrize("status", ["unavailable", "timeout", "invalid_response", "model_missing"])
def test_model_failure_returns_no_operational_action(monkeypatch, status):
    monkeypatch.setattr(chat, "request_json", Mock(return_value=(status, None)))
    turn = chat.reply_to_turn("今天有很多事想说")
    assert (turn.action, turn.source, turn.status) == ("reply", "fallback", status)


def test_chat_does_not_reenter_listening_mode_for_already_listening_user(monkeypatch):
    monkeypatch.setattr(chat, "request_json", Mock(side_effect=[
        ("ok", {"action": "listen", "quote": "我把报告带来了"}),
        ("ok", {"reply": "你愿意从报告的哪一部分说起？"}),
    ]))
    assert chat.reply_to_turn("我把报告带来了", listening=True).action == "reply"


def test_message_validation_and_generation_failure(monkeypatch):
    request = Mock(side_effect=[("ok", {"action": "reply", "quote": ""}), ("ok", {"reply": []})])
    monkeypatch.setattr(chat, "request_json", request)
    assert chat.reply_to_turn("").status == "empty"
    assert chat.reply_to_turn("啊" * (chat.MAX_MESSAGE_CHARS + 1)).status == "too_long"
    request.assert_not_called()
    assert chat.reply_to_turn("迷迷糊糊的一天").status == "invalid_response"


def test_listening_does_not_offer_to_fix_the_speakers_problem(monkeypatch):
    monkeypatch.setattr(chat, "request_json", Mock(side_effect=[
        ("ok", {"action": "reply", "quote": ""}), ("ok", {"reply": "需要我帮你整理开场白的要点吗？"})]))
    turn = chat.reply_to_turn("我怕一上台就忘了开场白。", listening=True)
    assert (turn.action, turn.source, turn.status) == ("reply", "local", "listening_guard")
    assert "帮你" not in turn.message


def test_explicit_recall_is_grounded_in_user_history_and_honors_corrections(monkeypatch):
    history = [("user", "我给小船取名红叶。"), ("assistant", "红叶或青禾？"),
               ("user", "刚说错了，最后用了银杏。")]
    lookup = Mock(side_effect=[("ok", {"action": "reply", "quote": ""}), ("ok", {"quote": "银杏"}),
                               ("ok", {"action": "reply", "quote": ""}), ("ok", {"quote": "青禾"})])
    monkeypatch.setattr(chat, "request_json", lookup)
    query = "还记得我最后用哪个名字吗？"
    assert "银杏" in chat.reply_to_turn(query, history, listening=True).message
    assert "红叶或青禾？" not in lookup.call_args.args[1]["past_user_messages"]
    assert chat.reply_to_turn(query, history).status == "unverified_recall"
