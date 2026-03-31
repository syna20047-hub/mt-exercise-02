#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Directory containing the experiment log files. 
LOG_DIR = Path("logs")
# Directory where all generated analysis files will be stored.
OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)


def parse_value(x: str):
    """
    Convert a string from the TSV log into a more useful Python value.

    Rules:
    - "final" is kept as a string because it represents a special marker
      rather than a numeric epoch/batch value.
    - values containing "." are parsed as floats
    - otherwise values are parsed as integers when possible
    - if parsing fails, the original string is returned

    This is mainly used for fields such as `epoch` and `batch`, which can
    contain either integers or special labels like "final".
    """
    if x == "final":
        return x
    try:
        if "." in x:
            return float(x)
        return int(x)
    except ValueError:
        return x


def load_logs(log_dir: Path):
    """
    Load and normalize all experiment rows from TSV log files.

    Each matching file (`log_*.tsv`) is read row by row, and selected fields
    are converted to appropriate Python types:
    - epoch, batch -> int / float / "final"
    - loss, ppl, dropout -> float

    A `source_file` field is also added so the origin of each row is preserved.

    Returns:
        list[dict]: A flat list of log rows from all input files.
    """
    rows = []
    # Read all log files in deterministic order so results are reproducible.
    for path in sorted(log_dir.glob("log_*.tsv")):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Normalize important columns into numeric values where possible.               
                row["epoch"] = parse_value(row["epoch"])
                row["batch"] = parse_value(row["batch"])
                row["loss"] = float(row["loss"])
                row["ppl"] = float(row["ppl"])
                row["dropout"] = float(row["dropout"])
                row["source_file"] = path.name
                rows.append(row)
    return rows


def collect(rows):
    """
    Group log rows into structures that are convenient for analysis.

    Returns three mappings:
        train_epoch: dropout -> {epoch: train perplexity}
        valid_epoch: dropout -> {epoch: validation perplexity}
        test_final:  dropout -> final test perplexity

    Only integer epochs are kept for train/validation curves, so rows with
    labels like "final" are ignored there. Test rows are assumed to already
    represent the final evaluation for each dropout value.
    """
    train_epoch = defaultdict(dict)   # dropout -> {epoch: ppl}
    valid_epoch = defaultdict(dict)   # dropout -> {epoch: ppl}
    test_final = {}                   # dropout -> ppl

    for row in rows:
        split = row["split"]
        dropout = row["dropout"]
        epoch = row["epoch"]
        ppl = row["ppl"]
        # Training perplexity measured at the end of each epoch.
        if split == "train_epoch" and isinstance(epoch, int):
            train_epoch[dropout][epoch] = ppl
        # Validation perplexity measured after each epoch.
        elif split == "valid" and isinstance(epoch, int):
            valid_epoch[dropout][epoch] = ppl
        # Final test perplexity for a completed model.
        elif split == "test":
            test_final[dropout] = ppl

    return train_epoch, valid_epoch, test_final


def all_epochs(*maps):
    """
    Collect all integer epoch numbers appearing in one or more epoch maps.

    Args:
        *maps: One or more dictionaries of the form
               dropout -> {epoch: value}

    Returns:
        list[int]: Sorted list of all distinct integer epochs.
    """
    epochs = set()
    for m in maps:
        for inner in m.values():
            epochs.update(inner.keys())
    # Keep only integer epochs to avoid special labels such as "final".
    return sorted(e for e in epochs if isinstance(e, int))


def write_epoch_table(path: Path, title: str, epoch_map: dict):
    """
    Write a TSV table for epoch-based perplexity values.

    The output format is:
        epoch    dropout_0.0    dropout_0.1    ...

    Each row corresponds to one epoch, and each column corresponds to one
    dropout setting. Missing values are written as empty strings.

    Args:
        path: Output TSV path.
        title: Header label for the first column (usually "epoch").
        epoch_map: Mapping of dropout -> {epoch: ppl}.
    """
    dropouts = sorted(epoch_map.keys())
    epochs = all_epochs(epoch_map)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        # Header row: epoch label followed by one column per dropout setting.
        writer.writerow([title] + [f"dropout_{d:.1f}" for d in dropouts])
        for epoch in epochs:
            row = [epoch]
            for d in dropouts:
                value = epoch_map[d].get(epoch, "")
                row.append(f"{value:.6f}" if value != "" else "")
            writer.writerow(row)


def write_test_table(path: Path, test_final: dict):
    """
    Write a TSV table containing the final test perplexity for each dropout.

    Args:
        path: Output TSV path.
        test_final: Mapping of dropout -> final test perplexity.
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dropout", "test_ppl"])
        for d in sorted(test_final.keys()):
            writer.writerow([f"{d:.1f}", f"{test_final[d]:.6f}"])


def write_markdown_summary(path: Path, train_epoch: dict, valid_epoch: dict, test_final: dict):
    """
    Write a short markdown summary of the experiment.

    Currently the summary focuses on final test perplexity and highlights:
    - the best dropout setting (lowest test perplexity)
    - the worst dropout setting (highest test perplexity)

    Args:
        path: Output markdown file path.
        train_epoch: Training perplexity by epoch (currently unused in text).
        valid_epoch: Validation perplexity by epoch (currently unused in text).
        test_final: Final test perplexity by dropout.
    """
    best_dropout = min(test_final, key=test_final.get)
    worst_dropout = max(test_final, key=test_final.get)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Dropout experiment summary\n\n")
        f.write("## Final test perplexity\n\n")
        f.write("| Dropout | Test ppl |\n")
        f.write("|---|---:|\n")
        for d in sorted(test_final.keys()):
            f.write(f"| {d:.1f} | {test_final[d]:.2f} |\n")

        f.write("\n")
        f.write(f"- Best dropout: **{best_dropout:.1f}** with test ppl **{test_final[best_dropout]:.2f}**\n")
        f.write(f"- Worst dropout: **{worst_dropout:.1f}** with test ppl **{test_final[worst_dropout]:.2f}**\n")


def make_plot(path: Path, title: str, ylabel: str, epoch_map: dict):
    """
    Create and save a line plot of perplexity over epochs.

    A separate line is drawn for each dropout setting.

    Args:
        path: Output image path.
        title: Plot title.
        ylabel: Label for the y-axis.
        epoch_map: Mapping of dropout -> {epoch: value}.
    """
    plt.figure(figsize=(8, 5))
    for dropout in sorted(epoch_map.keys()):
        epochs = sorted(epoch_map[dropout].keys())
        values = [epoch_map[dropout][e] for e in epochs]
        plt.plot(epochs, values, marker="o", label=f"dropout={dropout:.1f}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(sorted(all_epochs(epoch_map)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_console_summary(test_final: dict):
    """
    Print a compact summary of final test perplexities to the console.
    Args:
        test_final: Mapping of dropout -> final test perplexity.
    """
    best_dropout = min(test_final, key=test_final.get)
    worst_dropout = max(test_final, key=test_final.get)

    print("Final test perplexities:")
    for d in sorted(test_final.keys()):
        print(f"  dropout={d:.1f}: {test_final[d]:.2f}")
    print()
    print(f"Best model : dropout={best_dropout:.1f}, test ppl={test_final[best_dropout]:.2f}")
    print(f"Worst model: dropout={worst_dropout:.1f}, test ppl={test_final[worst_dropout]:.2f}")


def main():
    """
    Run the full analysis pipeline.

    Steps:
    1. Load all log rows from `logs/`
    2. Collect train/validation/test perplexities by dropout
    3. Write TSV tables
    4. Create plots
    5. Write a markdown summary
    6. Print a console summary
    """
    rows = load_logs(LOG_DIR)
    if not rows:
        raise SystemExit("No log files found in logs/")

    train_epoch, valid_epoch, test_final = collect(rows)

    write_epoch_table(OUT_DIR / "train_perplexity_table.tsv", "epoch", train_epoch)
    write_epoch_table(OUT_DIR / "valid_perplexity_table.tsv", "epoch", valid_epoch)
    write_test_table(OUT_DIR / "test_perplexity_table.tsv", test_final)

    make_plot(
        OUT_DIR / "train_perplexity_plot.png",
        "Training Perplexity over Epochs",
        "Perplexity",
        train_epoch,
    )
    make_plot(
        OUT_DIR / "valid_perplexity_plot.png",
        "Validation Perplexity over Epochs",
        "Perplexity",
        valid_epoch,
    )

    write_markdown_summary(
        OUT_DIR / "summary.md",
        train_epoch,
        valid_epoch,
        test_final,
    )

    print_console_summary(test_final)
    print("\nSaved files in:", OUT_DIR.resolve())
    print("- train_perplexity_table.tsv")
    print("- valid_perplexity_table.tsv")
    print("- test_perplexity_table.tsv")
    print("- train_perplexity_plot.png")
    print("- valid_perplexity_plot.png")
    print("- summary.md")


if __name__ == "__main__":
    main()