use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeDecision {
    pub symbol: String,
    pub drop_pct: f64,
    pub action: String,
    pub order_id: Option<String>,
    pub price_at_decision: f64,
    pub timestamp: String,
}

/// Loads trade decisions from JSON, validates them in parallel, and sorts the
/// accepted records by timestamp.
pub fn load_and_validate(path: &Path) -> Result<Vec<TradeDecision>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let records: Vec<TradeDecision> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {} as JSON", path.display()))?;

    if records.is_empty() {
        return Ok(Vec::new());
    }

    let (tx, rx) = mpsc::channel();
    let worker_count = MAX_WORKERS.min(records.len());
    let chunk_size = (records.len() + worker_count - 1) / worker_count;
    let mut handles = Vec::new();

    for chunk in records.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let tx = tx.clone();
        handles.push(thread::spawn(move || {
            for record in chunk {
                if is_valid(&record) {
                    let _ = tx.send(Some(record));
                } else {
                    eprintln!("dropping invalid record: {:?}", record);
                    let _ = tx.send(None);
                }
            }
        }));
    }
    drop(tx);

    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("pipeline worker thread panicked"))?;
    }

    let mut valid: Vec<TradeDecision> = rx.into_iter().flatten().collect();
    valid.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    Ok(valid)
}

const MAX_WORKERS: usize = 4;

fn is_valid(r: &TradeDecision) -> bool {
    !r.symbol.trim().is_empty()
        && !r.action.trim().is_empty()
        && !r.timestamp.trim().is_empty()
        && r.price_at_decision > 0.0
}