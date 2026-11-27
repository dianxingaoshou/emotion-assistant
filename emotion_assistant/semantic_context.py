"""Conservative Chinese transcript annotations for voluntary need confirmation."""
import re


def chinese_context_rules(text):
    line = text.strip()[:512]
    quote = bool(re.search(r'[“”「」『』"‘’]|朗读|引用|台词|念一[句段]|照着读|原话|[他她]说', line))
    other = re.search(r"[他她你]|别人|朋友|同学|同事|家人|哥哥|姐姐|弟弟|妹妹|妈妈|爸爸", line)
    subject = "other" if other else "self" if "我" in line and "我们" not in line else "unknown"
    if other and re.search(r"我也|但我|而我", line):
        subject = "unknown"
    if re.search(r"如果|假如|要是|可能会|明天|将来|以后|万一", line):
        scope = "hypothetical"
    elif re.search(r"昨天|前天|刚才|过去|上周|曾经|以前|上次|之前", line):
        scope = "past"
    elif re.search(r"现在|此刻|眼下|今天|我(?:觉得|感到|有点|很|好累|想|卡住)", line):
        scope = "current"
    else:
        scope = "unknown"
    feeling = re.search(r"烦躁|紧张|难受|疲惫|很累|好累|焦虑|平静|开心", line)
    # Avoid inferring wishes from reported speech, questions or negated utterances.
    denied = bool(re.search(r"并不|不是|没有|没觉得|不想|不再|不太|不需要|无需|不用|不必|不打算|别|"
                            r"不(?:紧张|难受|开心|累|烦躁|焦虑)|[？?吗]", line))
    need = None
    if subject == "self" and scope == "current" and not quote and not denied:
        for pattern, key in ((r"卡住|解决不了|不知道怎么开始", "stuck"),
                             (r"放松|紧张|身体紧绷", "tense"), (r"反复想|停不下思绪", "rumination"),
                             (r"想休息|疲惫|很累|好累", "fatigue")):
            if re.search(pattern, line):
                need = key
                break
    return {"version": "zh-context-rules-v1", "subject": subject, "time_scope": scope,
            "is_quote": quote, "explicit_feeling": feeling.group() if feeling and not denied else None,
            "need_hint": need, "evidence_text": line, "requires_confirmation": True}


def interpret_context(text):
    context = chinese_context_rules(text)
    # Store derived text data under the same consent and deletion boundary as ASR.
    from .english_polarity import predict_english_polarity
    from .english_intensity import predict_english_intensity

    polarity = predict_english_polarity(text)
    if polarity is not None:
        context["english_polarity"] = polarity
    intensity = predict_english_intensity(text)
    if intensity is not None:
        context["english_intensity"] = intensity
    from .expression_profile import text_expression_profile

    profile = text_expression_profile(text)
    if profile is not None:
        context["expression_profile"] = profile
    return context
