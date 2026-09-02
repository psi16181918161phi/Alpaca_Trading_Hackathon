mod pipeline;
mod serialization;

use std::path::Path;

const INPUT_PATH: &str = "../../memory_log.json";
const OUTPUT_PATH: &str = "trade_decisions.parquet";

fn main() -> anyhow::Result<()> {
    let input_path = Path::new(INPUT_PATH);
    let output_path = Path::new(OUTPUT_PATH);

    println!("reading {}", input_path.display());
    let records = pipeline::load_and_validate(input_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load and validate {}: {error}",
            input_path.display()
        )
    })?;
    println!("loaded {} valid trade decisions", records.len());

    if records.is_empty() {
        println!(
            "nothing to write, {} is empty or missing",
            input_path.display()
        );
        return Ok(());
    }

    serialization::write_parquet(&records, output_path)
        .map_err(|error| anyhow::anyhow!("failed to write {}: {error}", output_path.display()))?;
    println!("wrote {} rows to {}", records.len(), output_path);

    Ok(())
}
