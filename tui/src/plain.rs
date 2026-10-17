//! Plain-text rendering: what `--plain` prints, what CI checks, and what the replay test reads.

use crate::runs::{Run, Runs};
use std::fmt::Write;

pub fn render(runs: &Runs) -> String {
    let mut out = String::new();
    for run in &runs.runs {
        render_run(&mut out, run);
        out.push('\n');
    }
    out.push_str(&summary(runs));
    out.push('\n');
    out
}

pub fn header(run: &Run) -> String {
    let mut parts = vec![run.title()];
    if !run.identity_id.is_empty() {
        parts.push(format!("{} {}", run.action, run.identity_id));
    }
    parts.extend(run.model.clone());
    parts.extend(run.case_type.clone());
    parts.join(" · ")
}

pub fn verdict_line(run: &Run) -> String {
    let mut line = match (&run.report, &run.error) {
        (Some(r), _) => format!("=> {} ({})", r.verdict, r.confidence),
        (None, Some(e)) => format!("=> run failed: {e}"),
        (None, None) => "=> running…".to_string(),
    };
    if let (Some(expected), Some(ok)) = (&run.expected, run.matches_expected()) {
        let mark = if run.unsafe_delete() {
            "UNSAFE DELETE"
        } else if ok {
            "✓"
        } else {
            "✗"
        };
        let _ = write!(line, " · expected {expected} {mark}");
    }
    if let Some(r) = &run.report {
        let _ = write!(line, " · {} steps", r.steps);
        let cited: Vec<_> = r
            .findings
            .iter()
            .flat_map(|f| f.evidence_ids.iter().map(String::as_str))
            .collect();
        if !cited.is_empty() {
            let _ = write!(line, " · cites {}", cited.join(" "));
        }
        if r.rejected_citations > 0 {
            let _ = write!(line, " · {} citations dropped", r.rejected_citations);
        }
    }
    line
}

fn render_run(out: &mut String, run: &Run) {
    let _ = writeln!(out, "{}", header(run));
    for s in &run.steps {
        let _ = writeln!(out, "  [{}] {:<6} {}", s.n, s.node, s.text);
    }
    let _ = writeln!(out, "  {}", verdict_line(run));
    if let Some(reason) = run
        .report
        .as_ref()
        .and_then(|r| r.incomplete_reason.as_deref())
    {
        let _ = writeln!(out, "     incomplete: {reason}");
    }
}

pub fn summary(runs: &Runs) -> String {
    let n = runs.runs.len();
    let matched = runs
        .runs
        .iter()
        .filter(|r| r.matches_expected() == Some(true))
        .count();
    let incomplete = runs
        .runs
        .iter()
        .filter(|r| r.verdict() == "insufficient_evidence")
        .count();
    let mut s = format!(
        "{n} runs · {matched} match expected · {incomplete} incomplete · {} unsafe deletes",
        runs.unsafe_deletes()
    );
    if runs.bad_lines > 0 {
        let _ = write!(s, " · {} unreadable lines skipped", runs.bad_lines);
    }
    s
}
