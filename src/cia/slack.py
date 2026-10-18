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
    from .chat import Chat
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


def reply(text: str, desk: Desk) -> tuple[str, list[dict[str, Any]]]:
    """Answer an @mention from the read-only desk: (fallback text, Block Kit blocks).

    The person is found in code (request id, identity id or full name), so Slack text
    never reaches the model; everything shown is escaped. Approving stays on the
    change message's buttons."""
    question = MENTION.sub("", text).strip()
    if not question:
        return HELP, [_section(HELP)]
    ids = desk.identities_in(question)
    by_identity = desk.changes_by_identity()
    if not ids and WHO_LEFT.search(question):
        return who_left_card(desk.leavers())
    if not ids:
        note = f"I found no person in that question.\n{HELP}"
        return note, [_section(note)]
    blocks: list[dict[str, Any]] = []
    for i in ids[:3]:  # a card is about 15 blocks; Slack allows 50
        if blocks:
            blocks.append({"type": "divider"})
        blocks += card(desk, i, by_identity.get(i, []))
    names = ", ".join(desk.name(i) for i in ids[:3])
    return f"What I know about {names}", blocks


def _section(md: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": md[:3000]}}


def _context(md: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": md[:3000]}]}


def _bullets(title: str, lines: list[str], limit: int = 6) -> dict[str, Any]:
    shown = [f"• {escape(x)}" for x in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"_+{len(lines) - limit} more_")
    return _section(f"*{escape(title)}*\n" + "\n".join(shown))


STATUS_ICON = {"active": "🔴", "disabled": "🟠", "deleted": "⚫"}


def who_left_card(rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if not rows:
        return "HR marks nobody as gone.", [_section("HR marks nobody as gone.")]
    still = sum(r["account_status"] == "active" for r in rows)
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{len(rows)} people have left"},
        },
        _context(
            f"🔴 {still} accounts still active · 🟠 disabled · ⚫ deleted · "
            "ask `@Change Impact Agent <name>` for the full profile"
        ),
    ]
    lines = [
        f"{STATUS_ICON.get(r['account_status'], '•')} *{escape(r['name'])}* · "
        f"{escape(r.get('title') or (r['department'] or '-'))} · "
        f"last day {escape(r.get('last_day') or '?')} · _{escape(r['state'])}_"
        for r in rows
    ]
    for i in range(0, len(lines), 10):  # keep each section well under 3000 characters
        blocks.append(_section("\n".join(lines[i : i + 10])))
    return f"{len(rows)} people have left, {still} accounts still active", blocks


def card(desk: Desk, identity_id: str, rids: list[str]) -> list[dict[str, Any]]:
    """One person or service as a card: who, warnings, access, credentials, activity,
    what depends on it, tickets; then each change the agent made for them."""
    p = desk.profile(identity_id)
    if p is None:
        return [_section(f"No identity {escape(identity_id)}.")]
    icon = STATUS_ICON.get(p["account"], "⚪")
    gone = (
        f"left {p['last_day']} · " if p["hr_status"] == "left" and p["last_day"] else ""
    )
    account = "STILL ACTIVE" if p["account"] == "active" else p["account"]
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{icon} {p['name']}"[:150]},
        },
        _context(
            f"{escape(p['id'])} · {p['kind']} · {escape(gone)}account *{account}*"
        ),
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*{k}*\n{escape(str(v))}"}
                for k, v in p["fields"]
            ],
        },
    ]
    if p["warnings"]:
        blocks.append(_section("\n".join(f"⚠️ {escape(w)}" for w in p["warnings"])))
    for title, lines in p["sections"]:
        blocks.append(_bullets(title, lines))
    if not rids:
        blocks.append(
            _context(
                "_No offboarding change for this person yet: no HR event has started one._"
            )
        )
    for rid in rids:
        for change in desk.status(rid)["changes"]:
            verdict = (change.get("who") or "").rsplit(" · ", 1)[-1]
            lines = [
                f"*Agent* · {escape(verdict or 'waiting to start')} · `{escape(rid)}`"
            ]
            lines += [f"• {escape(f)}" for f in change.get("findings", [])]
            lines.append(f"*State:* {escape(change['state'])}")
            by = [
                e["by"]
                for e in desk.audit(rid)["entries"]
                if e["action"] == "approval" and e.get("by")
            ]
            if by:
                lines.append(f"*Answered by:* {escape(by[-1])}")
            blocks += [{"type": "divider"}, _section("\n".join(lines))]
    return blocks


def chat_reply(
    event: dict[str, Any], desk: Desk, chat: Chat | None
) -> tuple[str, list[dict[str, Any]]]:
    """An @mention answered by the chat model when there is one (its answer, then the
    person's card when it is about one person), else by the cards alone."""
    question = MENTION.sub("", event.get("text", "")).strip()
    if chat is None or not question:
        return reply(event.get("text", ""), desk)
    thread = f"{event.get('channel', '')}:{event.get('thread_ts') or event.get('ts')}"
    try:
        out = chat.answer(thread, question)
    except Exception as e:  # noqa: BLE001 - the model is down: still answer, from code
        text, blocks = reply(event.get("text", ""), desk)
        note = f"_The chat model is not answering ({type(e).__name__}); here is what I know._"
        return text, [_context(note), *blocks]
    blocks = [_section(out["text"])]
    if len(out["profiled"]) == 1:
        ident = out["profiled"][0]
        card_blocks = card(desk, ident, desk.changes_by_identity().get(ident, []))
        if len(blocks) + len(card_blocks) < 49:
            blocks += [{"type": "divider"}, *card_blocks]
    return out["text"], blocks


def socket_mode(
    bot_token: str,
    app_token: str,
    approvers: set[str],
    answer: Callable[[str, list[str], str], dict[str, Any]],
    desk: Desk | None = None,
    chat: Chat | None = None,
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
            text, blocks = chat_reply(event, desk, chat)
            say(
                text=text,
                blocks=blocks,
                thread_ts=event.get("thread_ts") or event.get("ts"),
            )

    handler = SocketModeHandler(app, app_token)
    handler.connect()  # returns at once; the websocket runs on its own thread
    return app, handler
