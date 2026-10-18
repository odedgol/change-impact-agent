//! Events grouped into runs. Each step names the graph node that produced it, the same names as
//! `src/cia/graph.py` and the monitor's Flow tab: model, nudge, tools, check, cap; and for the
//! change workflow (`src/cia/workflow.py`): guard, approve, verify, dry_run, disable, delete, restore.

use crate::events::{Event, Report};

pub const SAFE: &str = "safe_to_delete";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Turn,
    Nudge,
    Evidence,
    Refused,
    Retry,
    Cap,
    Gate,
    Wait,
    Act,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub n: u32,
    pub node: &'static str,
    pub kind: Kind,
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Run {
    pub run_id: Option<String>,
    pub model: Option<String>,
    pub case_type: Option<String>,
    pub expected: Option<String>,
    pub request_id: Option<String>,
    pub identity_id: String,
    pub action: String,
    /// What the requester wrote, shown above the steps.
    pub description: String,
    pub steps: Vec<Step>,
    pub report: Option<Report>,
    pub error: Option<String>,
    /// What the change workflow finally did: deleted, restored, kept_disabled, blocked, declined, failed.
    pub outcome: Option<String>,
    pub outcome_reason: Option<String>,
}

impl Run {
    pub fn title(&self) -> String {
        self.request_id
            .clone()
            .or_else(|| self.run_id.clone())
            .unwrap_or_else(|| "run".into())
    }

    pub fn finished(&self) -> bool {
        self.report.is_some() || self.error.is_some()
    }

    /// The verdict, or what stands in for one while the run is going or after it failed.
    pub fn verdict(&self) -> &str {
        match (&self.report, &self.error) {
            (Some(r), _) => &r.verdict,
            (None, Some(_)) => "error",
            (None, None) => "running",
        }
    }

    /// A delete recommended where the ground truth says something depends on the identity.
    pub fn unsafe_delete(&self) -> bool {
        self.verdict() == SAFE && self.expected.as_deref().is_some_and(|e| e != SAFE)
    }

    pub fn matches_expected(&self) -> Option<bool> {
        let report = self.report.as_ref()?;
        Some(self.expected.as_deref()? == report.verdict)
    }
}

/// All runs of a stream, in order. A `run` header starts a run; so does a second `request` without one.
#[derive(Debug, Default)]
pub struct Runs {
    pub runs: Vec<Run>,
    pub bad_lines: usize,
}

impl Runs {
    pub fn push_line(&mut self, line: &str) {
        match crate::events::parse_line(line) {
            None => {}
            Some(Ok(event)) => self.push(event),
            Some(Err(_)) => self.bad_lines += 1,
        }
    }

    pub fn push(&mut self, event: Event) {
        match event {
            Event::Run {
                run_id,
                model,
                case_type,
                expected_verdict,
            } => self.runs.push(Run {
                run_id,
                model,
                case_type,
                expected: expected_verdict,
                ..Run::default()
            }),
            Event::Request {
                request_id,
                identity_id,
                action,
                description,
            } => {
                // The workflow may investigate the same request again; that stays one run.
                let again = self
                    .runs
                    .last()
                    .is_some_and(|r| r.request_id.as_deref() == Some(request_id.as_str()));
                if !again && self.runs.last().is_none_or(|r| r.request_id.is_some()) {
                    self.runs.push(Run::default());
                }
                let run = self.current();
                run.request_id = Some(request_id);
                run.identity_id = identity_id;
                run.action = action;
                if !description.is_empty() {
                    run.description = description;
                }
            }
            Event::ModelTurn {
                step,
                reasoning,
                text,
                stop_reason,
                tool_calls,
            } => {
                let names: Vec<_> = tool_calls.iter().map(|c| c.name.as_str()).collect();
                let thought = reasoning.or(text).filter(|s| !s.is_empty());
                let (node, kind, mut line) = if names.is_empty() {
                    (
                        "nudge",
                        Kind::Nudge,
                        "no tool call: costs a step, gets a reminder".to_string(),
                    )
                } else {
                    ("model", Kind::Turn, format!("calls {}", names.join(", ")))
                };
                if stop_reason.as_deref() == Some("length") {
                    line.push_str(" (cut off: output limit)");
                }
                if let Some(t) = thought {
                    line = format!("{line}  · {}", one_line(&t, 90));
                }
                self.step(step, node, kind, line);
            }
            Event::ToolCall {
                step,
                tool,
                evidence_id,
            } => self.step(
                step,
                "tools",
                Kind::Evidence,
                format!("{tool} → {evidence_id}"),
            ),
            Event::ToolError { step, tool, error } => {
                let node = if tool == "submit_report" {
                    "check"
                } else {
                    "tools"
                };
                self.step(
                    step,
                    node,
                    Kind::Refused,
                    format!("{tool}: {}", one_line(&error, 90)),
                )
            }
            Event::CitationRetry { step, rejected_ids } => self.step(
                step,
                "check",
                Kind::Retry,
                format!("sent back once: unknown ids {}", rejected_ids.join(", ")),
            ),
            Event::StepCap { steps } => {
                // A turn with no tool call at the cap goes straight to the cap node, not to nudge.
                if let Some(last) = self.current().steps.last_mut() {
                    if last.kind == Kind::Nudge && last.n == steps {
                        last.node = "model";
                        last.text = last.text.replace(
                            "no tool call: costs a step, gets a reminder",
                            "no tool call",
                        );
                    }
                }
                self.step(
                    steps,
                    "cap",
                    Kind::Cap,
                    format!("step cap after {steps} steps → insufficient_evidence"),
                )
            }
            Event::RunError { error } => self.current().error = Some(error),
            Event::Report(report) => self.current().report = Some(report),
            Event::Guardrail { allowed, reasons } => {
                let text = if allowed.is_empty() {
                    format!("blocks: {}", reasons.join("; "))
                } else {
                    format!("allows {}", allowed.join(", "))
                };
                self.workflow_step("guard", Kind::Gate, text)
            }
            Event::ApprovalRequested { verdict, allowed } => self.workflow_step(
                "approve",
                Kind::Wait,
                format!("waits for a person ({verdict}): {}", allowed.join(", ")),
            ),
            Event::Approval {
                by,
                approved,
                refused,
            } => {
                let (kind, text) = match refused {
                    Some(r) => (Kind::Refused, format!("{by} asked for more: {r}")),
                    None if approved.is_empty() => (Kind::Gate, format!("{by} declined")),
                    None => (Kind::Gate, format!("{by} approved {}", approved.join(", "))),
                };
                self.workflow_step("approve", kind, text)
            }
            Event::EvidenceChanged { tools } => self.workflow_step(
                "verify",
                Kind::Retry,
                format!(
                    "evidence changed ({}) → investigate again",
                    tools.join(", ")
                ),
            ),
            Event::Action { action, outcome } => {
                let node = match action.as_str() {
                    "plan" => "dry_run",
                    "disable" => "disable",
                    "delete" => "delete",
                    "restore" => "restore",
                    _ => "action",
                };
                let text = match outcome {
                    Some(o) => format!("{action}: {o}"),
                    None => "plan only, nothing changed".to_string(),
                };
                self.workflow_step(node, Kind::Act, text)
            }
            Event::Outcome { outcome, reason } => {
                let run = self.current();
                run.outcome = Some(outcome);
                run.outcome_reason = reason;
            }
            Event::Unknown => {}
        }
    }

    fn current(&mut self) -> &mut Run {
        if self.runs.is_empty() {
            self.runs.push(Run::default());
        }
        self.runs.last_mut().expect("just pushed")
    }

    /// Workflow steps carry no step number; they follow the last one.
    fn workflow_step(&mut self, node: &'static str, kind: Kind, text: String) {
        let n = self.current().steps.last().map_or(0, |s| s.n);
        self.step(n, node, kind, text)
    }

    fn step(&mut self, n: u32, node: &'static str, kind: Kind, text: String) {
        self.current().steps.push(Step {
            n,
            node,
            kind,
            text,
        });
    }

    pub fn unsafe_deletes(&self) -> usize {
        self.runs.iter().filter(|r| r.unsafe_delete()).count()
    }
}

/// The first line of `s`, at most `max` characters.
pub fn one_line(s: &str, max: usize) -> String {
    let first = s.lines().next().unwrap_or("").trim();
    if first.chars().count() <= max {
        first.to_string()
    } else {
        format!("{}…", first.chars().take(max - 1).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(lines: &[&str]) -> Runs {
        let mut runs = Runs::default();
        lines.iter().for_each(|l| runs.push_line(l));
        runs
    }

    #[test]
    fn a_request_without_a_run_header_still_starts_a_run() {
        let runs = feed(&[
            r#"{"type":"request","request_id":"REQ-1","identity_id":"u1","action":"delete"}"#,
            r#"{"type":"report","verdict":"disable_first","confidence":"medium","findings":[],"steps":3}"#,
            r#"{"type":"request","request_id":"REQ-2","identity_id":"u2","action":"delete"}"#,
        ]);
        assert_eq!(runs.runs.len(), 2);
        assert_eq!(runs.runs[0].verdict(), "disable_first");
        assert_eq!(runs.runs[1].verdict(), "running");
    }

    #[test]
    fn steps_name_their_graph_node() {
        let runs = feed(&[
            r#"{"type":"run","run_id":"r","expected_verdict":"ask_owner"}"#,
            r#"{"type":"model_turn","step":0,"tool_calls":[]}"#,
            r#"{"type":"tool_error","step":1,"tool":"submit_report","error":"invalid report"}"#,
            r#"{"type":"citation_retry","step":2,"rejected_ids":["ev-x-9"]}"#,
            r#"{"type":"step_cap","steps":8}"#,
        ]);
        let nodes: Vec<_> = runs.runs[0].steps.iter().map(|s| s.node).collect();
        assert_eq!(nodes, ["nudge", "check", "check", "cap"]);
    }

    #[test]
    fn a_delete_against_the_ground_truth_is_unsafe() {
        let runs = feed(&[
            r#"{"type":"run","expected_verdict":"ask_owner"}"#,
            r#"{"type":"report","verdict":"safe_to_delete","findings":[]}"#,
            "garbage",
        ]);
        assert_eq!(runs.unsafe_deletes(), 1);
        assert_eq!(runs.bad_lines, 1);
    }

    #[test]
    fn one_line_cuts_long_text() {
        assert_eq!(one_line("abcdef\nsecond", 4), "abc…");
        assert_eq!(one_line("ok", 4), "ok");
    }
}
