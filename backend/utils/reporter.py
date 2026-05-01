"""
Rich-powered console reporter for ingestion results.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from ingestion.models import IngestionReport, Severity

console = Console()

SEVERITY_STYLE = {
    Severity.ERROR:   ("red",    "✖"),
    Severity.WARNING: ("yellow", "⚠"),
    Severity.INFO:    ("blue",   "ℹ"),
}


def print_report(report: IngestionReport) -> None:
    console.print()

    # ── Header ────────────────────────────────────────────────────────────────
    status_color = "green" if report.success and not report.has_errors else "red"
    status_label = "SUCCESS" if report.success else "FAILED"
    console.print(Panel(
        f"[bold {status_color}]{status_label}[/] — [dim]{report.file_path}[/]",
        title="[bold]Text Intelligence — Ingestion Report[/]",
        border_style="bright_black",
        expand=False,
        padding=(0, 2),
    ))

    if report.error:
        console.print(f"\n[bold red]Error:[/] {report.error}\n")
        return

    # ── Column Detection ──────────────────────────────────────────────────────
    det = report.text_column
    conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(det.confidence, "white")
    console.print(f"\n[bold]Text Column Detection[/]")
    console.print(f"  Column  : [bold cyan]{det.column_name}[/]")
    console.print(f"  Method  : [dim]{det.method}[/]")
    console.print(f"  Confidence: [{conf_color}]{det.confidence}[/]")
    console.print(f"  Reason  : [dim]{det.reasoning}[/]")

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    s = report.stats
    console.print(f"\n[bold]Dataset Statistics[/]")

    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column("Metric", style="dim", min_width=24)
    stats_table.add_column("Value", style="bold")

    stats_table.add_row("Total rows",        str(s.total_rows))
    stats_table.add_row("Total columns",     str(s.total_columns))
    stats_table.add_row("Clean rows",        f"[green]{s.clean_count}[/green]")
    stats_table.add_row("Null text values",  f"[{'red' if s.null_count else 'green'}]{s.null_count}[/] ({s.null_ratio:.1%})")
    stats_table.add_row("Empty (whitespace)", f"[{'yellow' if s.empty_count else 'green'}]{s.empty_count}[/]")
    stats_table.add_row("Too short",         f"[{'yellow' if s.too_short_count else 'green'}]{s.too_short_count}[/]")
    stats_table.add_row("Too long",          f"[{'yellow' if s.too_long_count else 'green'}]{s.too_long_count}[/]")
    stats_table.add_row("Duplicates",        f"[dim]{s.duplicate_count}[/]")
    stats_table.add_row("Avg text length",   f"{s.avg_text_length:.0f} chars")
    stats_table.add_row("Median text length",f"{s.median_text_length:.0f} chars")

    console.print(stats_table)

    if s.sources:
        console.print("  Sources:")
        for src, cnt in s.sources.items():
            console.print(f"    [dim]{src}[/dim]: {cnt}")

    # ── Issues ────────────────────────────────────────────────────────────────
    if report.issues:
        console.print(f"\n[bold]Validation Issues[/] ({len(report.issues)} found)\n")
        for issue in report.issues:
            style, icon = SEVERITY_STYLE[issue.severity]
            console.print(f"  [{style}]{icon} [{issue.severity.value.upper()}][/{style}] {issue.message}")
            if issue.row_indices and len(issue.row_indices) <= 10:
                console.print(f"    [dim]Rows: {issue.row_indices}[/]")
            elif issue.row_indices:
                console.print(f"    [dim]Rows (first 10): {issue.row_indices[:10]} ... (+{len(issue.row_indices)-10} more)[/]")
    else:
        console.print("\n[green]✔ No validation issues found.[/green]")

    # ── Output Paths ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]Saved Outputs[/]")
    if report.clean_csv_path:
        console.print(f"  Clean CSV  : [dim]{report.clean_csv_path}[/]")
    if report.report_json_path:
        console.print(f"  JSON report: [dim]{report.report_json_path}[/]")
    console.print()