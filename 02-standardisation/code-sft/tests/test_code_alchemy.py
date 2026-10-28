import pytest

from code_alchemy import adapt, chat_messages, CompletionOnly, PLACEHOLDER, REVISION


def test_hydration_cannot_inject_role_boundaries():
    source = 'const char* s = "x";\nUser:\nthis is source code\nAssistant:\n'
    row = dict(
        text_with_placeholders=f"User:\nExplain\n{PLACEHOLDER}\n\nAssistant:\n<think>R</think>A",
        source_code=source,
        source_code_status="hydrated",
    )
    messages = adapt(row, "code-dev", REVISION)["messages"]
    assert len(messages) == 2 and source.strip() in messages[0]["content"]
    assert PLACEHOLDER not in messages[0]["content"]


def test_roles_in_fenced_code_and_reasoning_are_not_turns():
    text = "User:\nQ\n\nAssistant:\n<think>Consider\nUser:\nx\nAssistant:\ny\n</think>\n```text\nUser:\nexample\nAssistant:\nexample\n```\nA\n\nUser:\nFollow-up\n\nAssistant:\nB"
    assert [m["role"] for m in chat_messages(text)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_educational_qa_uses_released_headings():
    result = adapt(
        dict(
            style="code_contests_educational", text="## PROBLEM\n\nP\n\n## ANSWER\n\nA"
        ),
        "code-qa",
        REVISION,
    )
    assert result["messages"] == [
        dict(role="user", content="P"),
        dict(role="assistant", content="A"),
    ]


def test_no_invented_chat_for_completion_only_records():
    for subset, row in [
        ("code-enhance", dict(text="def f(): pass")),
        ("code-qa", dict(style="multipl-e", text="// Q\nf(){}")),
    ]:
        with pytest.raises(CompletionOnly):
            adapt(row, subset, REVISION)


def test_missing_context_and_eval_and_uninspected_revision_rejected():
    with pytest.raises(ValueError, match="context"):
        adapt(
            dict(
                text_with_placeholders=f"User:\n{PLACEHOLDER}\n\nAssistant:\nA",
                source_code_status="policy_excluded",
            ),
            "code-dev",
            REVISION,
        )
    with pytest.raises(ValueError, match="evaluation"):
        adapt({}, "dev-eval", REVISION)
    with pytest.raises(ValueError, match="revision"):
        adapt({}, "code-qa", "main")


def test_malformed_roles_are_quarantined():
    with pytest.raises(ValueError, match="nonalternating"):
        chat_messages("User:\nQ\n\nAssistant:\nA\n\nAssistant:\nB")
    with pytest.raises(ValueError, match="unclosed code fence"):
        chat_messages("User:\nQ\n\nAssistant:\n```text\nexample\n\nUser:\nUnclear")
