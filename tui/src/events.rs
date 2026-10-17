//! The investigator's JSON-lines events (`cia.smoke.EVENT_TYPES`), parsed leniently.
//!
//! The UI only reads; it never imports the agent. Unknown event types and unknown fields are
//! ignored, so a newer agent never breaks an older UI. A line that is not JSON is skipped and counted.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Run {
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        case_type: Option<String>,
        #[serde(default)]
        expected_verdict: Option<String>,
    },
    Request {
        request_id: String,
        #[serde(default)]
        identity_id: String,
        #[serde(default)]
        action: String,
    },
    ModelTurn {
        #[serde(default)]
        step: u32,
        #[serde(default)]
        reasoning: Option<String>,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        stop_reason: Option<String>,
        #[serde(default)]
        tool_calls: Vec<CallName>,
    },
    ToolCall {
        #[serde(default)]
        step: u32,
        tool: String,
        evidence_id: String,
    },
    ToolError {
        #[serde(default)]
        step: u32,
        tool: String,
        error: String,
    },
    CitationRetry {
        #[serde(default)]
        step: u32,
        #[serde(default)]
        rejected_ids: Vec<String>,
    },
    StepCap {
        #[serde(default)]
        steps: u32,
    },
    RunError {
        error: String,
    },
    Report(Report),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CallName {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Report {
    pub verdict: String,
    #[serde(default)]
    pub confidence: String,
    #[serde(default)]
    pub findings: Vec<Finding>,
    #[serde(default)]
    pub rejected_citations: u32,
    #[serde(default)]
    pub incomplete_reason: Option<String>,
    #[serde(default)]
    pub steps: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Finding {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

/// One line of the stream: `None` for a blank line, `Some(Err)` for a line that is not an event.
pub fn parse_line(line: &str) -> Option<Result<Event, serde_json::Error>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    Some(serde_json::from_str(line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_report_and_ignores_unknown_fields() {
        let e = parse_line(
            r#"{"type":"report","verdict":"ask_owner","confidence":"high","findings":[],"new_field":1,"t":1.0}"#,
        );
        match e {
            Some(Ok(Event::Report(r))) => assert_eq!(r.verdict, "ask_owner"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unknown_types_and_blank_lines_are_harmless() {
        assert_eq!(
            parse_line(r#"{"type":"something_new","x":1}"#)
                .unwrap()
                .unwrap(),
            Event::Unknown
        );
        assert!(parse_line("   ").is_none());
        assert!(parse_line("not json").unwrap().is_err());
    }
}
