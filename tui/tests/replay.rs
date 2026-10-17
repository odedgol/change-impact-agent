//! Replays a recorded smoke stream (`python -m cia.smoke`) and checks what the UI shows.

use cia_tui::{plain, runs::Runs, ui};
use ratatui::{backend::TestBackend, Terminal};

const SMOKE: &str = include_str!("fixtures/smoke.jsonl");

fn replay() -> Runs {
    let mut runs = Runs::default();
    SMOKE.lines().for_each(|l| runs.push_line(l));
    runs
}

#[test]
fn every_line_of_the_recorded_run_is_understood() {
    let runs = replay();
    assert_eq!(runs.bad_lines, 0);
    assert_eq!(runs.runs.len(), 9);
    assert!(runs.runs.iter().all(|r| r.finished()));
}

#[test]
fn verdicts_match_the_ground_truth_and_nothing_unsafe_is_deleted() {
    let runs = replay();
    let verdicts: Vec<_> = runs
        .runs
        .iter()
        .map(|r| (r.title(), r.verdict().to_string()))
        .collect();
    assert_eq!(
        verdicts,
        [
            ("REQ-0001", "disable_first"),
            ("REQ-0003", "ask_owner"),
            ("REQ-0005", "disable_first"),
            ("REQ-0007", "ask_owner"),
            ("REQ-0009", "ask_owner"),
            ("REQ-0011", "ask_owner"),
            ("REQ-0013", "safe_to_delete"),
            ("REQ-0004", "ask_owner"),
            ("REQ-0017", "insufficient_evidence"),
        ]
        .map(|(a, b)| (a.to_string(), b.to_string()))
    );
    assert_eq!(runs.unsafe_deletes(), 0);
}

#[test]
fn the_guardrail_runs_show_every_graph_node() {
    let text = plain::render(&replay());
    for expected in [
        "[1] tools  delete_identity: unknown tool 'delete_identity'",
        "[5] check  sent back once: unknown ids ev-made-up-9",
        "[7] nudge  no tool call",
        "[8] model  no tool call",
        "[8] cap    step cap after 8 steps → insufficient_evidence",
        "=> ask_owner (high) · expected ask_owner ✓ · 6 steps · cites ev-deps-1",
        "9 runs · 8 match expected · 1 incomplete · 0 unsafe deletes",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn the_terminal_view_draws_the_selected_runs_verdict() {
    let mut app = ui::App::default();
    SMOKE.lines().for_each(|l| app.push_line(l));
    app.input_done = true;
    app.select(-1); // the guardrails run, one above the last
    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|f| ui::draw(f, &app)).unwrap();
    let screen: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    for expected in [
        "Runs",
        "REQ-0017",
        "insufficient_evidence",
        "sent back once",
        "=> ask_owner (high)",
        "replay",
    ] {
        assert!(screen.contains(expected), "screen is missing {expected:?}");
    }
}
