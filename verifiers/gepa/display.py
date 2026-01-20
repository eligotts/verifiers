"""
Simple terminal display for GEPA optimization.

Shows:
1. Budget progress bar (metric calls used)
2. Current phase/step indicator
3. Per-valset-row pareto frontier (best score for each row) - only from full valset evals
"""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TaskID
from rich.table import Table


@dataclass
class ValsetRowState:
    """Tracks best score for a single valset row."""
    best_score: float = 0.0
    best_candidate_idxs: list[int] = field(default_factory=list)


@dataclass
class GEPADisplayState:
    """Minimal state for display."""
    max_metric_calls: int = 500
    metric_calls_used: int = 0
    iteration: int = 0
    phase: str = "initializing"
    perfect_score: float | None = None

    # Minibatch tracking
    minibatch_before: float | None = None  # Score before reflection
    minibatch_after: float | None = None   # Score after reflection
    minibatch_accepted: bool | None = None
    minibatch_skipped: bool = False

    # Per-valset-row tracking (only from full valset evals)
    valset_rows: dict[int, ValsetRowState] = field(default_factory=dict)
    num_valset_evals: int = 0


class GEPADisplay:
    """
    Simple terminal display for GEPA optimization.

    Implements GEPA's LoggerProtocol (log method).
    Call update_eval() from adapter to track progress.
    """

    def __init__(
        self,
        max_metric_calls: int = 500,
        valset_size: int = 50,
        valset_example_ids: list[int] | None = None,
        log_file: str | Path | None = None,
        perfect_score: float | None = None,
    ) -> None:
        self.state = GEPADisplayState(
            max_metric_calls=max_metric_calls,
            perfect_score=perfect_score,
        )
        self.valset_size = valset_size
        self.valset_example_ids: set[int] | None = set(valset_example_ids) if valset_example_ids else None
        self.log_file = Path(log_file) if log_file else None

        self.console = Console()
        self.live: Live | None = None

        # Progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Budget"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=self.console,
            expand=False,
        )
        self.budget_task: TaskID | None = None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the live display."""
        self.budget_task = self.progress.add_task(
            "budget",
            total=self.state.max_metric_calls,
            completed=0,
        )
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self.live.start()

    def stop(self) -> None:
        """Stop the live display and print final summary."""
        if self.live:
            self.live.stop()
            self.live = None
        self._print_final_summary()

    def log(self, message: str) -> None:
        """LoggerProtocol - receives GEPA log messages."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

        # Parse phase from messages
        if "Base program full valset score" in message:
            self.state.phase = "initial valset done"
        elif "Selected program" in message:
            if "Iteration" in message:
                try:
                    self.state.iteration = int(message.split("Iteration")[1].split(":")[0].strip())
                except (ValueError, IndexError):
                    pass
            self.state.phase = "selecting"
        elif "Proposed new text" in message:
            self.state.phase = "re-evaluating"
        # Note: accepted/rejected phase is now set in update_eval() when post-reflection scores come in

        if self.live:
            self.live.update(self._render())

    def update_eval(
        self,
        candidate_idx: int,
        scores: list[float],
        example_ids: list[int],
        capture_traces: bool = False,
    ) -> None:
        """
        Called by adapter after each evaluation to update progress.

        Args:
            candidate_idx: Which candidate was evaluated
            scores: Scores for each example
            example_ids: Which valset rows were evaluated
            capture_traces: True if this is a pre-reflection eval (baseline)
        """
        # Update budget
        self.state.metric_calls_used += len(scores)
        if self.budget_task is not None:
            self.progress.update(self.budget_task, completed=self.state.metric_calls_used)

        # Check if this is a valset eval by matching example_ids
        if self.valset_example_ids is not None:
            is_valset_eval = set(example_ids) == self.valset_example_ids
        else:
            is_valset_eval = len(scores) == self.valset_size

        if is_valset_eval:
            # Full valset evaluation - update frontier
            self.state.num_valset_evals += 1
            for example_id, score in zip(example_ids, scores):
                row = self.state.valset_rows.get(example_id)
                if row is None:
                    self.state.valset_rows[example_id] = ValsetRowState(
                        best_score=score,
                        best_candidate_idxs=[candidate_idx],
                    )
                elif score > row.best_score:
                    # New best - replace
                    row.best_score = score
                    row.best_candidate_idxs = [candidate_idx]
                elif score == row.best_score and candidate_idx not in row.best_candidate_idxs:
                    # Tie - add to list
                    row.best_candidate_idxs.append(candidate_idx)
        else:
            # Minibatch evaluation - track before/after for status display
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if capture_traces:
                # This is the baseline eval before reflection
                self.state.minibatch_before = avg_score
                self.state.minibatch_after = None
                self.state.minibatch_accepted = None
                
                # Detect skip based on perfect score
                if self.state.perfect_score is not None and avg_score >= self.state.perfect_score:
                    self.state.minibatch_skipped = True
                else:
                    self.state.minibatch_skipped = False
                    # Baseline done, now waiting for teacher LLM to reflect
                    self.state.phase = "reflecting"
            else:
                # This is the eval after reflection
                self.state.minibatch_after = avg_score
                if self.state.minibatch_before is not None:
                    self.state.minibatch_accepted = avg_score > self.state.minibatch_before
                    # Set phase based on result
                    self.state.phase = "accepted" if self.state.minibatch_accepted else "rejected"

        if self.live:
            self.live.update(self._render())

    def _render(self) -> Group:
        """Render the full display."""
        return Group(
            self.progress,
            self._render_status(),
            self._render_frontier(),
        )

    def _render_status(self) -> Panel:
        """Render current status."""
        s = self.state
        phase_styles = {
            "initializing": "dim",
            "initial valset done": "green",
            "selecting": "blue",
            "reflecting": "magenta",
            "re-evaluating": "cyan",
            "accepted": "bold green",
            "rejected": "red",
        }
        style = phase_styles.get(s.phase, "white")

        # Build status line
        parts = [
            f"[bold]Iteration:[/bold] {s.iteration}",
            f"[bold]Phase:[/bold] [{style}]{s.phase}[/]",
        ]

        # Add minibatch info
        if s.minibatch_before is not None:
            if s.minibatch_skipped:
                parts.append(
                    f"[bold]Minibatch:[/bold] {s.minibatch_before:.2f} "
                    f"[bold green]✓ perfect (skipped reflection)[/]"
                )
            elif s.minibatch_after is None:
                # Waiting for post-reflection eval
                parts.append(f"[bold]Minibatch:[/bold] {s.minibatch_before:.2f} → ...")
            elif s.minibatch_accepted:
                # Accepted - show improvement
                parts.append(
                    f"[bold]Minibatch:[/bold] {s.minibatch_before:.2f} → "
                    f"[bold green]{s.minibatch_after:.2f} ✓ accepted[/]"
                )
            else:
                # Rejected - show decline
                parts.append(
                    f"[bold]Minibatch:[/bold] {s.minibatch_before:.2f} → "
                    f"[red]{s.minibatch_after:.2f} ✗ rejected[/]"
                )

        content = "  ".join(parts)
        return Panel(content, title="[bold]Status[/]", border_style="blue")

    def _render_frontier(self) -> Panel:
        """Render per-valset-row best scores (only from full valset evals)."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Row", style="cyan", width=6, justify="right")
        table.add_column("Best", style="green", width=6, justify="right")
        table.add_column("Prompt#", style="yellow", justify="left")

        rows = self.state.valset_rows
        if not rows:
            table.add_row("-", "-", "[dim]Waiting for valset eval...[/]")
        else:
            for row_idx in sorted(rows.keys()):
                row = rows[row_idx]
                score_style = "green" if row.best_score >= 1.0 else "yellow" if row.best_score > 0 else "red"
                prompts_str = ",".join(str(idx) for idx in row.best_candidate_idxs)
                table.add_row(
                    str(row_idx),
                    f"[{score_style}]{row.best_score:.2f}[/]",
                    prompts_str,
                )

        # Show summary line
        if rows:
            title = f"[bold]Valset Frontier[/] ({self.state.num_valset_evals} evals)"
        else:
            title = "[bold]Valset Frontier[/]"

        return Panel(table, title=title, border_style="green")

    def _print_final_summary(self) -> None:
        """Print final summary."""
        s = self.state
        self.console.print()
        self.console.print("[bold blue]" + "═" * 50)
        self.console.print("[bold blue]GEPA Complete")
        self.console.print("[bold blue]" + "═" * 50)

        self.console.print(f"\n[bold]Budget:[/bold] {s.metric_calls_used}/{s.max_metric_calls}")
        self.console.print(f"[bold]Iterations:[/bold] {s.iteration}")
        self.console.print(f"[bold]Full valset evals:[/bold] {s.num_valset_evals}")
