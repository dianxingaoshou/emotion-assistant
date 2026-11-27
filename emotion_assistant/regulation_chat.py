"""One turn of an opt-in local support conversation. Only the UI executes transitions."""

from dataclasses import dataclass
import re
import unicodedata

from .local_qwen import request_json
from .regulation import ACTIVITIES


@dataclass(frozen=True)
class ChatTurn:
    action: str
    message: str
    source: str
    status: str = "ok"


MAX_MESSAGE_CHARS = 1000
HISTORY_MESSAGES = 8
_CONTROL_REPLIES = {
    "stop": "好，先停在这里。想继续时可以再告诉我。",
    "listen": "好，我听着。你可以从最想说的地方说起。",
    "switch": "好，我们看看还有哪些可选的方式。由你决定要不要开始。",
    "discomfort": "先停下来。谢谢你告诉我你的感受。",
}
# Match direct requests as whole clauses, leaving quotes, negation and reports for contextual review.
_POLITE = r"(?:那|嗯|好|麻烦你|请你|请|我想要|我想|我希望|咱们|我们|可以|能不能|能否|现在|先)*"
_END = r"(?:行吗|好吗|好不好|可以吗|吗|吧|了|谢谢|啦|哦)*"
_CONTROLS = {
    "stop": r"停(?:止|下来|一停|一下|一会儿|一会)?(?:这个|当前的?)?(?:练习|活动|引导|聊天)?|"
            r"暂停(?:一下|练习|活动|引导)?|结束(?:这次|这段|当前)?(?:练习|活动|聊天|对话|引导)?|"
            r"(?:我)?不想(?:再)?继续(?:了|练习|活动|聊)?|到这里|别继续|别打扰我|stop|pause",
    "switch": r"(?:帮我|给我)?换(?:一个|一种|个|种|一下|成别的|别的)(?:其他|别的|新的)?"
              r"(?:方法|方式|活动|练习|办法)?|试试别的(?:方法|活动)?|换换方法|switch|try something else",
    "listen": r"(?:我)?(?:现在|暂时|今天)?(?:只想|就想|想先|只需要|只要)(?:和你|跟你)?"
              r"(?:聊一聊|聊聊天|聊会儿|聊聊|聊一会儿|说说话|说说|说两句|倾诉|找个人聊聊)|"
              r"(?:你)?(?:就|先)?听我说(?:说|一会儿)?|别给我(?:安排|推荐)(?:活动|练习)|just listen|just want to talk",
    "discomfort": r"(?:这个|这项|刚才的?)?(?:活动|练习|呼吸)(?:让|弄得|使)我(?:更烦|更难受|很难受|不舒服)|"
                  r"(?:我|我感觉|我觉得|感觉|觉得)?(?:有点|很|更)?(?:不舒服|难受)|this feels uncomfortable",
}
_DIRECT = {action: re.compile(_POLITE + "(?:" + body + ")" + _END, re.I)
           for action, body in _CONTROLS.items()}
_QUOTATIONS = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"]*"|‘[^’]*’')
_INTRODUCES_QUOTE = re.compile(r"(?:说(?:的是|过|道)?|告诉我|举个例子|比如|例如)[：:]?$")
_NEGATED_REQUEST = {
    "stop": r"(?:不要|不想|不需要|不必|别)(?:再|先)?(?:停|结束)",
    "switch": r"(?:不要|不想|不需要|不必|别)(?:再|先)?(?:换|改)",
    "listen": r"(?:不想|不要|不用|不需要)(?:和你|跟你)?(?:聊|你听我说|倾诉)",
    "discomfort": r"(?:没有|没|并不|不再)(?:觉得|感到|很)?(?:不舒服|难受)",
}
_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["reply", *_CONTROL_REPLIES]},
        "quote": {"type": "string", "maxLength": 150},
    },
    "required": ["action", "quote"], "additionalProperties": False,
}
_INTENT_PROMPT = """Identify the user's request in this single message, not their mood. Output JSON.
情绪表达、询问或回忆、引用、否定、假设、向别人提的要求用 reply。
stop 表示用户想停止当前引导或对话；switch 表示用户明确想换活动或从聊天转向选一种活动；
listen 表示只想倾诉不想练习，也包括要求有人听自己说、撤下建议以留出倾诉空间；
discomfort 表示当前练习让本人感到不适；其他一律 reply。
例：“今天压力很大”是 reply；“能陪我说说话吗”是 listen；“把这个环节结束吧”是 stop。
quote 为 current_message 中表达当前操作意愿的连续原话。current_message 仅是待分析的对话。"""
_REPLY_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string", "minLength": 1, "maxLength": 240}},
                 "required": ["reply"], "additionalProperties": False}
_REPLY_PROMPT = """你是桌面情绪线索助手。前面的消息是本次对话。请用对方的语言简短地回应最新一条。
如果用户问先前说过的事，从前面的用户消息找出具体答案，有纠正时采用更新后的说法。上下文确实缺少答案时再澄清。
回应用户主动讲的事，保留说话人和听者的关系。引用转述时先分清谁对谁说；不确定时省略转述。
可以引用具体细节，允许不提问，每次最多一个问题。倾听模式不主动推活动。
不要诊断、编造个人历史、猜测自评分或声称已操作界面。本人决定要不要开始练习。只输出 JSON 的 reply 字段。"""


def local_command(text):
    """Handle explicit controls immediately even if an outstanding model call has stalled."""
    if not isinstance(text, str) or not text.strip():
        return None
    unquoted = _QUOTATIONS.sub("［引文］", unicodedata.normalize("NFKC", text))
    intent = None
    for sentence in re.split(r"[。.!！?？;；\n]+", unquoted):
        previous = ""
        for clause in re.split(r"[，,]+", sentence):
            clause = clause.strip()
            if not _INTRODUCES_QUOTE.search(previous):
                for action, pattern in _DIRECT.items():
                    if pattern.fullmatch(clause):
                        if action == "discomfort":
                            return ChatTurn(action, _CONTROL_REPLIES[action], "local")
                        intent = action
                        break
            previous = clause
    if intent:
        return ChatTurn(intent, _CONTROL_REPLIES[intent], "local")
    return None


def _fallback(status):
    return ChatTurn("reply", "这次未能收到可用的千问回复。你可以重试，也可以直接说“停一下”“换一种”或“我只想聊聊”。",
                    "fallback", status)


_LISTEN_PROMPT = (
    "\n当前已进入用户明确要求的倾听模式。只简短复述听到的具体事情、回答前文问题，"
    "或邀对方继续讲自己的经历。不要提出解决办法、练习、建议或服务，也不要询问要不要帮忙。"
    "对方明确要换活动时，应用会单独处理。"
)
_ADVICE_OFFER = re.compile(
    r"(?:需要|要不要|要我|想让我|我可以|让我来|我来|不妨|建议你|你可以试试|我们可以试|咱们可以先)"
    r".{0,12}(?:帮|整理|练习|尝试|试试|计划|准备|列出|列个|做个|梳理|解决|呼吸|放松|拆解|建议)|"
    r"要不要(?:试|做|先)|我能帮你|不妨试|可以先试|"
    r"\b(?:would you like me to|shall (?:we|I)|do you want me to|you (?:could|should) try|I suggest)\b",
    re.I)

_RECALL = re.compile(r"(?:刚才|之前|前面|前边).{0,8}(?:我说|我提到|我讲|说过)|"
                     r"(?:还记得|记不记得).{0,16}(?:我|什么|哪个)|what did I (?:say|mention)", re.I)
_RECALL_SCHEMA = {"type": "object", "properties": {"quote": {"type": "string", "maxLength": 150}},
                  "required": ["quote"], "additionalProperties": False}


def reply_to_turn(text, history=(), *, listening=False, activity=None):
    """Synchronous worker entry. History is scoped to this conversation by its caller."""
    if not isinstance(text, str) or not text.strip():
        return ChatTurn("reply", "请先输入想说的话。", "local", "empty")
    text = text.strip()
    command = local_command(text)
    if command:
        return command
    if len(text) > MAX_MESSAGE_CHARS:
        return ChatTurn("reply", f"每条消息最多 {MAX_MESSAGE_CHARS} 个字，请分开发送。", "local", "too_long")

    recent = [{"role": role, "content": message[:MAX_MESSAGE_CHARS]}
              for role, message in history[-HISTORY_MESSAGES:]
              if role in ("user", "assistant") and isinstance(message, str)]
    activity_title = ACTIVITIES[activity]["title"] if activity in ACTIVITIES else None
    payload = {"current_message": text, "active_activity": activity_title}
    status, result = request_json(_INTENT_PROMPT, payload, _ACTION_SCHEMA, num_predict=192)
    if status != "ok":
        return _fallback(status)
    if not isinstance(result, dict) or set(result) != {"action", "quote"}:
        return _fallback("invalid_response")
    action, quote = result["action"], result["quote"]
    if not isinstance(action, str) or not isinstance(quote, str) or len(quote) > 150:
        return _fallback("invalid_response")
    if action != "reply":
        unquoted = _QUOTATIONS.sub("", text)
        if (action not in _CONTROL_REPLIES or not quote.strip() or quote not in unquoted
                or re.search(_NEGATED_REQUEST[action], unquoted)):
            return _fallback("invalid_response")
        if not ((action == "listen" and listening) or (action == "discomfort" and not activity_title)):
            return ChatTurn(action, _CONTROL_REPLIES[action], "qwen")
    # The router has no history to replay accidentally. The conversational answer uses native chat roles.
    if recent and _RECALL.search(text):
        past = [row["content"] for row in recent if row["role"] == "user"]
        status, result = request_json(
            "用户正在回想自己说过的事情。从 past_user_messages 逐字摘出直接回答 question 的具体词句。"
            "按先后顺序阅读，若本人后来纠正，取纠正后的表述。列表中没有答案就返回空 quote。",
            {"past_user_messages": past, "question": text}, _RECALL_SCHEMA)
        if status != "ok":
            return _fallback(status)
        quote = result.get("quote") if isinstance(result, dict) else None
        if isinstance(quote, str) and 0 < len(quote.strip()) <= 150 and any(quote in msg for msg in past):
            return ChatTurn("reply", f"你之前提到：“{quote}”。", "qwen")
        return ChatTurn("reply", "我暂时没找到对应的原话。你愿意再告诉我一次吗？", "local", "unverified_recall")
    prompt = _REPLY_PROMPT + _LISTEN_PROMPT if listening else _REPLY_PROMPT
    status, result = request_json(prompt, dict(payload, listen_only=bool(listening)),
                                  _REPLY_SCHEMA, num_predict=320, history=recent)
    if status != "ok":
        return _fallback(status)
    if not isinstance(result, dict) or set(result) != {"reply"}:
        return _fallback("invalid_response")
    reply = result["reply"]
    if not isinstance(reply, str) or not 0 < len(reply.strip()) <= 240:
        return _fallback("invalid_response")
    if listening and _ADVICE_OFFER.search(_QUOTATIONS.sub("", reply)):
        return ChatTurn("reply", "我在听。你可以接着说。", "local", "listening_guard")
    return ChatTurn("reply", reply.strip(), "qwen")
