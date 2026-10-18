"""Slack: the change message with buttons, and the answers that come back.

``SlackNotifier`` posts ``notify.message`` as Block Kit, updates it after an answer and
puts the outcome in its thread. ``answer_from_click`` turns a button click into
``Agent.answer``: only people on the approver allow-list may answer, and the buttons
only ever carry what the guardrails allowed (the workflow checks again anyway).

Socket Mode (``slack_bolt``) keeps one outbound websocket open, so no public URL is
needed: the same code runs on a laptop, in a private cloud or on AWS. Model text is
escaped before it reaches Slack, and nothing read in Slack ever reaches the model.

Settings (``.env``): SLACK_BOT_TOKEN (xoxb-), SLACK_APP_TOKEN (xapp-, connections:write),
CIA_SLACK_CHANNEL (C0...), CIA_SLACK_APPROVERS (U0...,U0...). App: infra/slack/manifest.yaml.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from .notify import Message

if TYPE_CHECKING:
    from .mcp_server import Desk

APPROVE, DECLINE = "cia_approve", "cia_decline"


class SlackClient(Protocol):
    """The two Web API calls we use (``slack_sdk.WebClient`` has them)."""

    def chat_postMessage(self, **kwargs: Any) -> Any: ...  # noqa: N802 - Slack's name
    def chat_update(self, **kwargs: Any) -> Any: ...


def escape(text: str) -> str:
    """Slack mrkdwn control characters; model text is shown, never interpreted."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def choices(actions: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """The buttons for the allowed actions: a delete always comes with its disable."""
    out = []
    if "disable" in actions:
        out.append(("Approve disable", ["disable"]))
    if "delete" in actions:
        out.append(("Approve disable + delete", ["disable", "delete"]))
    return out


def blocks(m: Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{escape(m.title)}*"}},
    ]
    if m.lines:
        out.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(f"• {escape(line)}" for line in m.lines),
                },
            }
        )
    out.append({"type": "section", "text": {"type": "mrkdwn", "text": escape(m.state)}})
    buttons: list[dict[str, Any]] = [
        {
            "type": "button",
            "action_id": f"{APPROVE}_{i}",
            "text": {"type": "plain_text", "text": label},
            "style": "primary" if i == 0 else None,
            "value": json.dumps({"request_id": m.request_id, "actions": acts}),
        }
        for i, (label, acts) in enumerate(choices(m.actions))
    ]
    if m.actions:
        buttons.append(
            {
                "type": "button",
                "action_id": DECLINE,
                "text": {"type": "plain_text", "text": "Decline"},
                "style": "danger",
                "value": json.dumps({"request_id": m.request_id, "actions": []}),
            }
        )
        out.append(
            {
                "type": "actions",
                "elements": [
                    {k: v for k, v in b.items() if v is not None} for b in buttons
                ],
            }
        )
    if m.link:
        out.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"<{m.link}|Full trace>"}],
            }
        )
    return out


class SlackNotifier:
    """Posts each change to one channel; the reference is ``channel:ts``."""

    def __init__(self, client: SlackClient, channel: str):
        self.client, self.channel = client, channel

    def post(self, message: Message) -> str:
        r = self.client.chat_postMessage(
            channel=self.channel, text=message.title, blocks=blocks(message)
        )
        return f"{r['channel']}:{r['ts']}"

    def update(self, ref: str, message: Message) -> None:
        channel, _, ts = ref.partition(":")
        if not ts:  # posted elsewhere (console); start a message here instead
            self.post(message)
            return
        self.client.chat_update(
            channel=channel, ts=ts, text=message.title, blocks=blocks(message)
        )
        if message.done:
            self.client.chat_postMessage(
                channel=channel, thread_ts=ts, text=escape(message.state)
            )


def answer_from_click(
    body: dict[str, Any],
    approvers: set[str],
    answer: Callable[[str, list[str], str], dict[str, Any]],
) -> str | None:
    """Handle one button click. Returns a note for the clicker (shown only to them), or
    None when the answer went through and the message itself will update."""
    user = body.get("user") or {}
    uid = user.get("id", "")
    if uid not in approvers:
        return "You are not on the approver list for changes (CIA_SLACK_APPROVERS)."
    try:
        value = json.loads(body["actions"][0]["value"])
        request_id, actions = str(value["request_id"]), list(value["actions"])
    except (KeyError, IndexError, TypeError, ValueError):
        return "This button is not one of ours."
    name = re.sub(
        r"[^A-Za-z0-9_.@ -]", "", user.get("username") or user.get("name") or uid
    )
    try:
        status = answer(request_id, actions, f"{name or uid} (slack {uid})")
    except (KeyError, ValueError) as e:  # already answered, or not waiting
        return f"Not applied: {e}"
    if status.get("outcome") == "declined" and actions:
        return f"Refused: {status.get('reason')}"
    return None


MENTION = re.compile(r"<@[A-Z0-9]+>")
WHO_LEFT = re.compile(r"\bwho\b.*\b(left|gone)\b|\blist\b|\bchanges\b", re.I)
HELP = (
    "Ask me: `@Change Impact Agent who left?` lists everyone HR marks as gone; "
    "`@Change Impact Agent what's going on with Noah Novak?` (a full name, an id "
    "like `u00034`, or a request `OFF-...`) shows the facts and the agent's verdict."
)


def reply(text: str, desk: Desk) -> str:
    """Answer an @mention from the read-only desk. The person is found in code (request
    id, identity id or full name), so Slack text never reaches the model; what comes
    back is escaped. Approving stays on the message's buttons."""
    question = MENTION.sub("", text).strip()
    if not question:
        return HELP
    ids = desk.people(question)
    agent_ids = desk.changes_by_identity()
    for rid in desk.mentioned(question):
        ident = next(i for i, r in agent_ids.items() if rid in r)
        if ident not in ids:
            ids.append(ident)
    if not ids and WHO_LEFT.search(question):
        rows = desk.leavers()
        if not rows:
            return "HR marks nobody as gone."
        head = f"*{len(rows)} people HR marks as gone:*"
        return "\n".join(
            [head]
            + [
                f"• *{escape(r['name'])}* · {escape(r['department'] or '-')} · "
                f"account {escape(r['account_status'])} · {escape(r['state'])}"
                for r in rows
            ]
        )
    if not ids:
        return f"I found no person in that question.\n{HELP}"
    return "\n\n".join(person(desk, i, agent_ids.get(i, [])) for i in ids[:5])


def person(desk: Desk, identity_id: str, rids: list[str]) -> str:
    """One person: the org's facts, then each change the agent made for them."""
    facts = desk.facts(identity_id)
    name = desk.name(identity_id)
    lines = [f"*{escape(name)}* ({escape(identity_id)})"]
    lines += [f"• {escape(f)}" for f in facts]
    if not rids:
        lines.append("_No offboarding change for this person yet._")
    for rid in rids:
        for change in desk.status(rid)["changes"]:
            lines.append(f"*Agent:* {escape(change.get('who') or rid)} · {escape(rid)}")
            lines += [f"   ◦ {escape(f)}" for f in change.get("findings", [])]
            lines.append(escape(change["state"]))
            by = [
                e["by"]
                for e in desk.audit(rid)["entries"]
                if e["action"] == "approval" and e.get("by")
            ]
            if by:
                lines.append(f"Answered by {escape(by[-1])}")
    return "\n".join(lines)


def socket_mode(
    bot_token: str,
    app_token: str,
    approvers: set[str],
    answer: Callable[[str, list[str], str], dict[str, Any]],
    desk: Desk | None = None,
):
    """Listen for button clicks, and for @mentions when a desk is given, over Socket
    Mode (a background websocket)."""
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=bot_token)

    @app.action(re.compile(f"^({APPROVE}_\\d+|{DECLINE})$"))
    def clicked(ack, body, respond):
        ack()
        note = answer_from_click(body, approvers, answer)
        if note:
            respond(text=note, response_type="ephemeral", replace_original=False)

    if desk is not None:

        @app.event("app_mention")
        def mentioned(event, say):
            say(
                text=reply(event.get("text", ""), desk),
                thread_ts=event.get("thread_ts") or event.get("ts"),
            )

    handler = SocketModeHandler(app, app_token)
    handler.connect()  # returns at once; the websocket runs on its own thread
    return app, handler
