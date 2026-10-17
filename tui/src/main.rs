//! cia-tui [FILE|-] [--plain]
//!
//! Reads JSON-lines events from FILE (or stdin, the default) and shows each run step by step, then
//! its colour-coded verdict with citations. On a terminal it is interactive and follows a live stream;
//! with --plain, or when stdout is not a terminal, it prints text and exits 1 if any run recommended
//! an unsafe delete.

use cia_tui::{plain, runs::Runs, ui};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::io::{self, BufRead, BufReader, IsTerminal, Read};
use std::sync::mpsc;
use std::time::Duration;
use std::{env, fs, process, thread};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cia-tui [FILE|-] [--plain]");
        return;
    }
    let plain_mode = args.iter().any(|a| a == "--plain") || !io::stdout().is_terminal();
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "-".into());
    let input: Box<dyn Read + Send> = if path == "-" {
        Box::new(io::stdin())
    } else {
        match fs::File::open(&path) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("cia-tui: cannot open {path}: {e}");
                process::exit(2);
            }
        }
    };

    if plain_mode {
        let mut runs = Runs::default();
        for line in BufReader::new(input).lines().map_while(Result::ok) {
            runs.push_line(&line);
        }
        print!("{}", plain::render(&runs));
        process::exit(if runs.unsafe_deletes() > 0 { 1 } else { 0 });
    }
    if let Err(e) = interactive(input) {
        eprintln!("cia-tui: {e}");
        process::exit(1);
    }
}

/// Lines arrive on a channel from a reader thread, so a live stream shows up as it is written.
fn interactive(input: Box<dyn Read + Send>) -> io::Result<()> {
    let (tx, rx) = mpsc::channel::<Option<String>>();
    thread::spawn(move || {
        for line in BufReader::new(input).lines().map_while(Result::ok) {
            if tx.send(Some(line)).is_err() {
                return;
            }
        }
        let _ = tx.send(None);
    });

    let mut terminal = ratatui::init();
    let mut app = ui::App::default();
    let result = loop {
        while let Ok(msg) = rx.try_recv() {
            match msg {
                Some(line) => app.push_line(&line),
                None => app.input_done = true,
            }
        }
        if let Err(e) = terminal.draw(|f| ui::draw(f, &app)) {
            break Err(e);
        }
        match event::poll(Duration::from_millis(100)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => app.select(1),
                    KeyCode::Up | KeyCode::Char('k') => app.select(-1),
                    _ => {}
                },
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => {}
            Err(e) => break Err(e),
        }
    };
    ratatui::restore();
    result
}
