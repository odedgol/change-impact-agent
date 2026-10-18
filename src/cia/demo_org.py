"""A demo org that looks like a real company: people with names, service accounts with
purposes, the same hidden traps.

The bench org (``cia.generator``) names people ``user-00034`` so that scores never depend
on names. For a demo that is hard to follow, so this module takes the same generated
org (120 identities: one of each planted case, eleven decoys) and dresses it: every
person gets a first and last name, every service account a name such as
``svc-billing-export``, and ids inside dependency names become the person's handle.
Ids, dependencies, usage, tickets and the ground truth's answers stay as they are, so
the agent faces exactly the same evidence. The ``lookalike_name`` case gets a decoy's
name with one letter swapped (l for I), as a real lookalike would.

    python -m cia.demo_org        # make demo-org: data/demo/org.db, data/demo/truth.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import timedelta
from pathlib import Path

from .generator import NOW, generate, offboarding_text

FIRST = [
    "Dana",
    "Noa",
    "Yael",
    "Maya",
    "Tamar",
    "Shira",
    "Roni",
    "Lior",
    "Omer",
    "Itai",
    "Amit",
    "Yonatan",
    "Daniel",
    "Eitan",
    "Ariel",
    "Gal",
    "Nadav",
    "Ido",
    "Ella",
    "Michal",
    "Hila",
    "Keren",
    "Avi",
    "Ben",
    "Adam",
    "Lea",
    "Sara",
    "Rachel",
    "Emma",
    "Olivia",
    "Liam",
    "Noah",
    "Lucas",
    "Mia",
    "Sofia",
    "Chloe",
    "Ethan",
    "Jacob",
    "Ava",
    "Isabella",
    "Aiden",
    "Leo",
    "Nina",
    "Priya",
    "Arjun",
    "Wei",
    "Mei",
    "Hiro",
    "Yuki",
    "Carlos",
    "Lucia",
    "Mateo",
    "Elena",
    "Omar",
    "Layla",
    "Fatima",
    "Ivan",
    "Olga",
]
LAST = [
    "Levi",
    "Cohen",
    "Mizrahi",
    "Peretz",
    "Biton",
    "Friedman",
    "Shapiro",
    "Katz",
    "Weiss",
    "Klein",
    "Rosen",
    "Stern",
    "Goldberg",
    "Ben-David",
    "Avraham",
    "Hadad",
    "Azoulay",
    "Dahan",
    "Malka",
    "Amar",
    "Smith",
    "Johnson",
    "Brown",
    "Garcia",
    "Martinez",
    "Lee",
    "Patel",
    "Kim",
    "Chen",
    "Wang",
    "Nguyen",
    "Silva",
    "Rossi",
    "Muller",
    "Novak",
    "Ivanova",
    "Tanaka",
    "Sato",
    "Kumar",
    "Singh",
    "Haddad",
    "Nasser",
    "Okafor",
    "Mensah",
    "Larsen",
    "Berg",
    "Dubois",
    "Moreau",
]
PURPOSE = [
    "billing-export",
    "crm-sync",
    "payroll-feed",
    "reports-nightly",
    "etl-orders",
    "backup-runner",
    "sso-bridge",
    "ticket-bot",
    "metrics-push",
    "invoice-mailer",
    "ledger-recon",
    "search-indexer",
    "data-lake-ingest",
    "github-actions",
    "deploy-bot",
    "jira-sync",
    "slack-alerts",
    "vendor-sftp",
    "hr-import",
    "access-review",
]
N_IDENTITIES = 120
OUT = Path("data/demo")
VERSION = "2"  # make demo rebuilds an org written by an older version (2: profiles)
DOMAIN = "acme.io"
LOCATIONS = ("Tel Aviv", "New York", "London", "Berlin", "Remote (EU)", "Remote (US)")
TITLES = {
    "engineering": ("Software Engineer", "Senior Backend Engineer", "Staff Engineer"),
    "data": ("Data Engineer", "Analytics Engineer", "Senior Data Scientist"),
    "finance": ("Financial Analyst", "Senior Accountant", "FP&A Manager"),
    "legal": ("Legal Counsel", "Senior Legal Counsel", "Paralegal"),
    "sales": ("Account Executive", "Sales Engineer", "Regional Sales Manager"),
    "support": ("Support Engineer", "Customer Success Manager", "Support Lead"),
    "security": ("Security Engineer", "Senior SecOps Engineer", "IAM Lead"),
    "hr": ("HR Business Partner", "Recruiter", "People Ops Manager"),
}
APPS = {
    "engineering": ("GitHub", "AWS Console", "Datadog", "PagerDuty", "Jira"),
    "data": ("Snowflake", "Looker", "AWS Console", "GitHub"),
    "finance": ("NetSuite", "Expensify", "Snowflake"),
    "legal": ("DocuSign", "Ironclad"),
    "sales": ("Salesforce", "Gong", "HubSpot"),
    "support": ("Zendesk", "Salesforce", "Jira"),
    "security": ("Okta Admin", "AWS Console", "CrowdStrike", "GitHub"),
    "hr": ("Workday", "Greenhouse"),
}
BASE_APPS = ("Google Workspace", "Slack", "Zoom", "Confluence")
AWS_ROLES = {
    "engineering": ("eng-developer", "eng-deploy"),
    "data": ("data-analyst", "data-pipeline-admin"),
    "finance": ("finance-readonly",),
    "security": ("security-audit", "break-glass-admin"),
}
REPOS = {
    "engineering": ("acme/api", "acme/web", "acme/infra"),
    "data": ("acme/pipelines", "acme/dbt-models"),
    "security": ("acme/infra", "acme/detections"),
}
AWS_ACCOUNT = "acme-prod (4411-2093-7781)"


def names(n: int, rng: random.Random) -> list[str]:
    pairs = [f"{f} {last}" for f in FIRST for last in LAST]
    rng.shuffle(pairs)
    return pairs[:n]


def lookalike(name: str) -> str:
    """One letter that reads the same: a lower-case l becomes a capital I, or an o a 0."""
    for a, b in (("l", "I"), ("o", "0"), ("a", "@")):
        if a in name:
            return name.replace(a, b, 1)
    return name + "."


def handle(name: str) -> str:
    return name.lower().replace(" ", ".")


def build(out: str | Path = OUT, seed: int = 7) -> dict:
    """Write ``out/org.db`` and ``out/truth.json``; returns the ground truth."""
    out = Path(out)
    db = out / "org.db"
    truth = generate(db, n_identities=N_IDENTITIES, seed=seed)
    rng = random.Random(seed)
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT id, kind FROM identities ORDER BY id").fetchall()
    humans = [i for i, kind in rows if kind == "human"]
    services = [i for i, kind in rows if kind == "service"]
    person = dict(zip(humans, names(len(humans), rng), strict=True))
    purposes = list(PURPOSE)
    rng.shuffle(purposes)
    service = {
        s: f"svc-{purposes[i % len(purposes)]}"
        + (f"-{i // len(purposes) + 1}" if i >= len(purposes) else "")
        for i, s in enumerate(services)
    }
    by_type = {c["case_type"]: c for c in truth["cases"]}
    decoy = next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")
    if "lookalike_name" in by_type:  # a decoy's name, one letter swapped
        person[by_type["lookalike_name"]["identity_id"]] = lookalike(
            person[decoy["identity_id"]]
        )
    conn.executemany(
        "UPDATE identities SET name = ? WHERE id = ?",
        [(n, i) for i, n in {**person, **service}.items()],
    )

    def renamed(text: str) -> str:
        """Ids inside a longer name (es-api-key-u00009) become the handle; a bare id stays."""
        for ident, name in person.items():
            if ident in text and text != ident:
                text = text.replace(ident, handle(name))
        return text

    deps = conn.execute("SELECT rowid, source, target_id FROM dependencies").fetchall()
    conn.executemany(
        "UPDATE dependencies SET source = ?, target_id = ? WHERE rowid = ?",
        [(renamed(src), renamed(tgt), rowid) for rowid, src, tgt in deps],
    )
    names_by_id = {**person, **service}
    for c in truth["cases"]:
        dept = conn.execute(
            "SELECT department FROM identities WHERE id = ?", (c["identity_id"],)
        ).fetchone()[0]
        c["name"] = names_by_id[c["identity_id"]]
        c["description"] = offboarding_text(c["name"], dept)
        for h in c["hidden_dependencies"]:
            h["ref"] = renamed(h["ref"])
    write_profiles(conn, random.Random(seed + 1))
    conn.commit()
    conn.close()
    truth["style"] = "demo"
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "VERSION").write_text(VERSION + "\n")
    return truth


def _key(rng: random.Random) -> str:
    return "AKIA" + "".join(
        rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)
    )


def _day(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).date().isoformat()


def write_profiles(conn: sqlite3.Connection, rng: random.Random) -> None:
    """A mocked HR record, access and credentials per identity, in their own table.

    The agent's tools never read ``profiles``: its evidence (usage, dependencies,
    tickets, owner) is the bench's, unchanged. The profile is what a person sees in
    Slack, so it is made to agree with that evidence: a leaver's last day is the HR
    notice's date, an account that a job or a pipeline uses has an access key.
    """
    conn.execute(
        "CREATE TABLE profiles (identity_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
    )
    rows = conn.execute(
        "SELECT id, kind, name, department, hr_status FROM identities ORDER BY id"
    ).fetchall()
    staff = [r for r in rows if r[1] == "human" and r[4] != "left"]
    for ident, kind, name, dept, hr_status in rows:
        dept = dept or "engineering"
        notice = conn.execute(
            "SELECT MIN(ts) FROM tickets WHERE identity_id = ? AND author = 'hr-bot'",
            (ident,),
        ).fetchone()[0]
        machine = (
            conn.execute(
                "SELECT COUNT(*) FROM usage_events WHERE identity_id = ? AND caller != 'interactive'",
                (ident,),
            ).fetchone()[0]
            or conn.execute(
                "SELECT COUNT(*) FROM dependencies WHERE target_id = ? AND kind IN "
                "('config_ref', 'owns_credential', 'runs_as')",
                (ident,),
            ).fetchone()[0]
        )
        keys = [
            {"id": _key(rng), "created": _day(rng.randint(200, 1200))}
            for _ in range(1 if machine or kind == "service" else 0)
        ]
        if kind == "service":
            data = {
                "type": "AWS IAM user"
                if rng.random() < 0.6
                else "Okta API service app",
                "purpose": name.removeprefix("svc-").replace("-", " "),
                "created": _day(rng.randint(300, 1500)),
                "aws_policies": [f"{name}-policy", "CloudWatchAgentServerPolicy"],
                "keys": keys,
                "mfa": None,
            }
        else:
            peers = [r for r in staff if r[3] == dept and r[0] != ident] or staff
            manager = rng.choice(peers)
            apps = list(BASE_APPS) + list(APPS.get(dept, ()))
            data = {
                "title": rng.choice(TITLES.get(dept, ("Specialist",))),
                "location": rng.choice(LOCATIONS),
                "email": f"{handle(name)}@{DOMAIN}",
                "manager": manager[2],
                "manager_id": manager[0],
                "joined": _day(rng.randint(250, 2800)),
                "employment": "contractor" if rng.random() < 0.12 else "full-time",
                "last_day": notice[:10]
                if notice
                else (_day(rng.randint(10, 45)) if hr_status == "left" else None),
                "apps": apps,
                "aws_roles": list(AWS_ROLES.get(dept, ())),
                "repos": list(REPOS.get(dept, ())),
                "groups": [f"{dept}-all"]
                + ([f"{dept}-admins"] if rng.random() < 0.25 else []),
                "keys": keys,
                "mfa": "off"
                if rng.random() < 0.1
                else rng.choice(("TOTP", "Okta Verify", "WebAuthn")),
                "password_age_days": rng.randint(5, 400),
            }
        conn.execute("INSERT INTO profiles VALUES (?, ?)", (ident, json.dumps(data)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the realistic demo org.")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    truth = build(a.out, a.seed)
    print(
        f"{N_IDENTITIES} identities, {len(truth['cases'])} people to offboard -> {a.out}/"
    )
    for c in sorted(truth["cases"], key=lambda c: c["case_type"]):
        print(f"  {c['case_type']:<26} {c['name']:<22} {c['identity_id']}")


if __name__ == "__main__":
    main()
