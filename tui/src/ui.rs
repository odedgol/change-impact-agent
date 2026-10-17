//! The interactive view: runs on the left, the selected run's steps on the right, its verdict below.

use crate::plain::{header, summary, verdict_line};
use crate::runs::{Kind, Run, Runs};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

#[derive(Default)]
pub struct App {
    pub runs: Runs,
    pub selected: usize,
    /// Keep the newest run selected while events stream in, until the user picks one.
    pub pinned: bool,
    pub input_done: bool,
}

impl App {
    pub fn push_line(&mut self, line: &str) {
        self.runs.push_line(line);
        if !self.pinned {
            self.selected = self.runs.runs.len().saturating_sub(1);
        }
    }

    pub fn select(&mut self, delta: isize) {
        let n = self.runs.runs.len();
        if n == 0 {
            return;
        }
        self.pinned = true;
        self.selected = self.selected.saturating_add_signed(delta).min(n - 1);
    }
}

pub fn verdict_color(verdict: &str) -> Color {
    match verdict {
        "safe_to_delete" => Color::Green,
        "disable_first" => Color::Yellow,
        "ask_owner" => Color::Magenta,
        "insufficient_evidence" => Color::Gray,
        "error" => Color::Red,
        _ => Color::Cyan,
    }
}

fn kind_color(kind: Kind) -> Color {
    match kind {
        Kind::Turn => Color::Blue,
        Kind::Nudge => Color::Yellow,
        Kind::Evidence => Color::Green,
        Kind::Refused | Kind::Cap => Color::Red,
        Kind::Retry => Color::Yellow,
    }
}

pub fn draw(frame: &mut Frame, app: &App) {
    let [top, body, keys] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let state = if app.input_done { "replay" } else { "live" };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " Change Impact Agent ",
                Style::new().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("· {state} · {}", summary(&app.runs))),
        ])),
        top,
    );
    frame.render_widget(
        Paragraph::new(" ↑/↓ or j/k: pick a run   q: quit").style(Style::new().fg(Color::DarkGray)),
        keys,
    );

    let [left, right] =
        Layout::horizontal([Constraint::Length(34), Constraint::Min(30)]).areas(body);
    let items: Vec<ListItem> = app
        .runs
        .runs
        .iter()
        .map(|r| {
            let mark = if r.unsafe_delete() { " !" } else { "" };
            ListItem::new(Line::from(vec![
                Span::raw(format!("{:<9} ", r.title())),
                Span::styled(
                    format!("{}{mark}", r.verdict()),
                    Style::new().fg(verdict_color(r.verdict())),
                ),
            ]))
        })
        .collect();
    let mut list_state = ListState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(Block::new().borders(Borders::ALL).title(" Runs "))
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
        left,
        &mut list_state,
    );

    match app.runs.runs.get(app.selected) {
        Some(run) => draw_run(frame, right, run),
        None => frame.render_widget(
            Paragraph::new("Waiting for events… pipe in the investigator's --events output, or a runs/*.jsonl file.")
                .block(Block::new().borders(Borders::ALL)),
            right,
        ),
    }
}

fn draw_run(frame: &mut Frame, area: ratatui::layout::Rect, run: &Run) {
    let [steps_area, report_area] =
        Layout::vertical([Constraint::Min(5), Constraint::Length(8)]).areas(area);
    let lines: Vec<Line> = run
        .steps
        .iter()
        .map(|s| {
            Line::from(vec![
                Span::styled(format!("{:>2} ", s.n), Style::new().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:<6} ", s.node),
                    Style::new()
                        .fg(kind_color(s.kind))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(s.text.clone()),
            ])
        })
        .collect();
    // Show the newest steps when they no longer fit.
    let inner = steps_area.height.saturating_sub(2) as usize;
    let scroll = lines.len().saturating_sub(inner) as u16;
    frame.render_widget(
        Paragraph::new(lines).scroll((scroll, 0)).block(
            Block::new()
                .borders(Borders::ALL)
                .title(format!(" {} ", header(run))),
        ),
        steps_area,
    );

    let color = verdict_color(run.verdict());
    let mut report = vec![Line::styled(
        verdict_line(run),
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    )];
    if let Some(r) = &run.report {
        if let Some(reason) = &r.incomplete_reason {
            report.push(Line::raw(format!("incomplete: {reason}")));
        }
        for f in &r.findings {
            report.push(Line::from(vec![
                Span::raw(format!("• {}: {} ", f.kind, f.summary)),
                Span::styled(
                    format!("[{}]", f.evidence_ids.join(" ")),
                    Style::new().fg(Color::Cyan),
                ),
            ]));
        }
    }
    frame.render_widget(
        Paragraph::new(report).wrap(Wrap { trim: true }).block(
            Block::new()
                .borders(Borders::ALL)
                .title(" Verdict ")
                .border_style(Style::new().fg(color)),
        ),
        report_area,
    );
}
