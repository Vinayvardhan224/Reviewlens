"""
ReviewLens Command-Line Interface (CLI).
Interactive terminal tool for analyzing single reviews or entire datasets with rich formatting.
"""

import sys
import os
import argparse
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure parent path is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reviewlens.pipeline import ReviewLensPipeline

console = Console(safe_box=True)


def display_single_analysis(analysis):
    """Render a single review analysis in a styled rich layout."""
    label = analysis.sentiment.label
    color = "green" if label == "POSITIVE" else ("red" if label == "NEGATIVE" else "yellow")

    console.print(Panel(
        f"[bold white]{analysis.original_text}[/bold white]\n\n"
        f"[bold]Sentiment:[/bold] [{color}]{label}[/{color}] (Score: {analysis.sentiment.compound:+.3f})\n"
        f"[bold]Predicted Rating:[/bold] {analysis.predicted_rating:.1f} / 5.0\n"
        f"[bold]Dominant Emotion:[/bold] {analysis.sentiment.primary_emotion or 'N/A'}\n"
        f"[bold]Confidence:[/bold] {analysis.sentiment.confidence:.0%}",
        title="[bold cyan]ReviewLens - Review Analysis[/bold cyan]",
        expand=False
    ))

    # Spam check badge
    if analysis.spam.is_spam:
        console.print(Panel(
            f"[bold red][!] FLAGGED AS SPAM / SUSPICIOUS (Score: {analysis.spam.spam_score:.2f})[/bold red]\n"
            f"[yellow]Reasons:[/yellow] " + ", ".join(analysis.spam.reasons),
            border_style="red"
        ))

    # Aspect mentions table
    if analysis.aspects:
        table = Table(title="Extracted Aspect-Based Sentiments", show_header=True, header_style="bold magenta")
        table.add_column("Aspect Category", style="cyan")
        table.add_column("Matched Term", style="white")
        table.add_column("Polarity", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Sentence Context", style="dim")

        for a in analysis.aspects:
            pol_col = "green" if a.sentiment_label == "POSITIVE" else ("red" if a.sentiment_label == "NEGATIVE" else "yellow")
            table.add_row(
                a.category,
                a.matched_term,
                f"[{pol_col}]{a.sentiment_label}[/{pol_col}]",
                f"{a.sentiment_score:+.2f}",
                a.sentence[:60] + ("..." if len(a.sentence) > 60 else "")
            )
        console.print(table)
    else:
        console.print("[dim]No specific product aspects explicitly mentioned.[/dim]")


def display_batch_results(result, title="Dataset Intelligence Report"):
    """Render comprehensive batch analytics in rich formatted panels and tables."""
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()

    # KPI summary cards
    kpi_table = Table.grid(padding=(0, 2))
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_column()

    nss = result.executive_summary.net_sentiment_score
    nss_col = "green" if nss >= 30 else ("yellow" if nss >= 0 else "red")

    kpi_table.add_row(
        Panel(f"[bold white]{result.total_reviews}[/bold white]\n[dim]Total Reviews[/dim]", border_style="blue"),
        Panel(f"[{nss_col}]{nss:+.1f}[/{nss_col}]\n[dim]Net Sentiment Score[/dim]", border_style=nss_col),
        Panel(f"[bold green]{result.executive_summary.csat_score:.1f}%[/bold green]\n[dim]Estimated CSAT[/dim]", border_style="green"),
        Panel(f"[bold red]{result.spam_reviews}[/bold red]\n[dim]Spam Flagged[/dim]", border_style="red"),
    )
    console.print(kpi_table)
    console.print()

    # Aspect Performance Table
    if result.aspect_summaries:
        aspect_table = Table(title="Product Aspect Performance Breakdown", show_header=True, header_style="bold blue")
        aspect_table.add_column("Aspect Category", style="cyan", width=25)
        aspect_table.add_column("Mentions", justify="right", width=10)
        aspect_table.add_column("Pos %", justify="right", style="green", width=10)
        aspect_table.add_column("Neg %", justify="right", style="red", width=10)
        aspect_table.add_column("NSS", justify="right", style="bold", width=10)
        aspect_table.add_column("Representative Customer Quote", style="dim")

        for a in result.aspect_summaries:
            top_quote = a.sample_positive_quotes[0] if a.net_sentiment_score >= 0 and a.sample_positive_quotes else (
                a.sample_negative_quotes[0] if a.sample_negative_quotes else "N/A"
            )
            aspect_table.add_row(
                a.category,
                str(a.total_mentions),
                f"{a.sentiment_ratio.get('positive', 0):.0f}%",
                f"{a.sentiment_ratio.get('negative', 0):.0f}%",
                f"{a.net_sentiment_score:+.0f}",
                top_quote[:50] + ("..." if len(top_quote) > 50 else "")
            )
        console.print(aspect_table)
        console.print()

    # SWOT Analysis
    swot = result.executive_summary
    swot_table = Table(title="Executive SWOT Matrix", show_header=True, header_style="bold yellow")
    swot_table.add_column("Strengths (S)", style="green", ratio=1)
    swot_table.add_column("Weaknesses (W)", style="red", ratio=1)
    swot_table.add_row(
        "\n".join(f"* {s}" for s in swot.strengths[:3]),
        "\n".join(f"* {w}" for w in swot.weaknesses[:3])
    )
    swot_table.add_column("Opportunities (O)", style="blue", ratio=1)
    swot_table.add_column("Threats (T)", style="magenta", ratio=1)
    console.print(swot_table)
    console.print()

    # Prioritized Recommendations
    if swot.actionable_recommendations:
        rec_table = Table(title="Prioritized Product & Engineering Action Plan", show_header=True, header_style="bold red")
        rec_table.add_column("Priority", style="bold", width=12)
        rec_table.add_column("Action Item", style="white", width=35)
        rec_table.add_column("Responsible Team", style="cyan", width=25)
        rec_table.add_column("Tactical Directive", style="dim")

        for r in swot.actionable_recommendations[:4]:
            rec_table.add_row(
                f"[{r['badge_color']}]{r['priority']}[/{r['badge_color']}]",
                r["title"],
                r["responsible_team"],
                r["recommended_action"]
            )
        console.print(rec_table)


def run_demo():
    """Execute built-in demo analyzing the sample headphones review dataset."""
    sample_file = Path(__file__).resolve().parent / "data" / "sample_headphones_reviews.csv"
    if not sample_file.exists():
        console.print(f"[red]Sample data not found at {sample_file}[/red]")
        return

    console.print(Panel.fit(
        "[bold cyan]ReviewLens[/bold cyan] - NLP Smart Product Review Analysis System\n"
        "[dim]Running end-to-end intelligence demo on wireless headphones reviews...[/dim]",
        border_style="cyan"
    ))

    df = pd.read_csv(sample_file)
    pipeline = ReviewLensPipeline()

    console.print("[dim]Analyzing reviews with sentiment, aspect extraction, and topic models...[/dim]")
    result = pipeline.analyze_dataset(df, text_column="text", rating_column="rating")
    display_batch_results(result, title="Smart Headphones Review Intelligence Report")


def main():
    parser = argparse.ArgumentParser(description="ReviewLens: NLP-Based Smart Product Review Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: demo
    subparsers.add_parser("demo", help="Run quick demo on sample product reviews")

    # Command: analyze-text
    cmd_text = subparsers.add_parser("analyze-text", help="Analyze a single review string")
    cmd_text.add_argument("text", type=str, help="Review text to analyze")
    cmd_text.add_argument("--rating", type=float, default=None, help="Optional star rating (1.0 to 5.0)")

    # Command: analyze-file
    cmd_file = subparsers.add_parser("analyze-file", help="Analyze CSV or JSON dataset")
    cmd_file.add_argument("filepath", type=str, help="Path to input file")
    cmd_file.add_argument("--text-col", default="text", help="Column name for review text (default: 'text')")
    cmd_file.add_argument("--rating-col", default="rating", help="Column name for star rating (default: 'rating')")
    cmd_file.add_argument("--export", type=str, default=None, help="Optional output JSON file path")

    args = parser.parse_args()

    if args.command == "demo" or not args.command:
        run_demo()
    elif args.command == "analyze-text":
        pipeline = ReviewLensPipeline()
        analysis = pipeline.analyze_single(text=args.text, rating=args.rating)
        display_single_analysis(analysis)
    elif args.command == "analyze-file":
        path = Path(args.filepath)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            return
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path)
        pipeline = ReviewLensPipeline()
        result = pipeline.analyze_dataset(df, text_column=args.text_col, rating_column=args.rating_col)
        display_batch_results(result, title=f"Analysis of {path.name}")
        if args.export:
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            console.print(f"[green]Saved full report to {args.export}[/green]")


if __name__ == "__main__":
    main()
