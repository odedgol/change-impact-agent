//! Terminal UI for the change-impact agent. It only reads the investigator's JSON-lines events
//! (from a file, or piped live from `--events -`); it never imports or calls the agent.

pub mod events;
pub mod plain;
pub mod runs;
pub mod ui;
