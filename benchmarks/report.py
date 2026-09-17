"""
Build the benchmark report, `benchmarks/results/report.html`, from the results JSON

    python -m benchmarks.report [--results-dir benchmarks/results] [--output report.html]

The report is one self-contained HTML file. Plotly is loaded from a CDN for the
interactive charts, and everything that carries a conclusion (the summary tiles, the
fastest-method table, the correctness table) is plain HTML rendered here, so the file
still says what it has to say with no network and no JavaScript.
"""

import argparse
import html
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchmarks.report")

# Fixed colour per implementation, in the categorical slot order of the design
# system's validated palette (light and dark steps of the same eight hues). The
# mapping is by implementation, never by rank, so a method keeps its colour in every
# chart and does not change when a filter drops another series
SERIES_COLOURS: dict[str, tuple[str, str]] = {
    "oxvox-kdtree": ("#2a78d6", "#3987e5"),
    "oxvox-voxel": ("#eb6834", "#d95926"),
    "oxvox-hybrid": ("#1baf7a", "#199e70"),
    "oxvox-voxel-c2": ("#eda100", "#c98500"),
    "oxvox-kiddo": ("#e87ba4", "#d55181"),
    "oxvox-graph": ("#008300", "#008300"),
    "scipy": ("#4a3aa7", "#9085e9"),
    "open3d": ("#e34948", "#e66767"),
}

# Order competitors appear in legends and tables
SERIES_ORDER = list(SERIES_COLOURS)

# The reference implementation speed-ups are quoted against
REFERENCE = "scipy"

# Blue sequential ramp steps used for the recall heatmap, light to dark
SEQUENTIAL_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """
    Read every results file in a directory

    Args:
        results_dir: Directory holding the JSON files written by `benchmarks.run`

    Returns:
        One parsed results dict per file, ordered by dataset then point count
    """
    groups = [
        json.loads(path.read_text())
        for path in sorted(results_dir.glob("*.json"))
        if path.name != "report.json"
    ]
    if not groups:
        raise SystemExit(f"no results found in {results_dir}")
    groups.sort(key=lambda group: (group["dataset"]["label"], group["dataset"]["num_points"]))
    return groups


def _format_seconds(seconds: float) -> str:
    """
    Human-readable duration for a table cell
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _format_count(value: float) -> str:
    """
    Human-readable point or query count for a table cell or axis label
    """
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def column_label(plan: dict[str, Any]) -> str:
    """
    Short label for the grid column a radius forms

    Synthetic radii are chosen to hit a density target, which is comparable across
    datasets and point counts, so the target names the column. The real scans' radii
    are physical lengths with no target, so the length names the column instead

    Args:
        plan: One entry of a results file's `radius_plans`

    Returns:
        The column label
    """
    if plan["density_target"] > 0:
        return f"~{plan['density_target']:g} per sphere"
    return f"r = {plan['radius']:g} m"


def column_sort_key(plan: dict[str, Any]) -> tuple[int, float]:
    """
    Sort key putting the density-target columns first, each group ascending

    Args:
        plan: One entry of a results file's `radius_plans`

    Returns:
        The sort key
    """
    if plan["density_target"] > 0:
        return (0, plan["density_target"])
    return (1, plan["radius"])


class ReportModel:
    """
    Everything the report shows, extracted from the raw results once

    The raw files carry one record per measurement (including builds, skips and
    correctness checks); the report needs them keyed by grid point, so the reshaping
    happens here and both the charts and the static tables read the same numbers
    """

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        """
        Args:
            groups: Parsed results files, as returned by `load_results`
        """
        self.groups = groups
        self.meta = groups[-1]["meta"]
        self.competitor_specs: dict[str, dict[str, Any]] = {}
        for group in groups:
            self.competitor_specs.update(group["competitors"])

        self.competitors = [key for key in SERIES_ORDER if key in self.competitor_specs]
        self.competitors += [key for key in self.competitor_specs if key not in SERIES_ORDER]
        self.exact_competitors = [
            key for key in self.competitors if self.competitor_specs[key]["exact"]
        ]

        self.dataset_labels = sorted({group["dataset"]["label"] for group in groups})
        self.descriptions = {
            group["dataset"]["label"]: group["dataset"]["description"] for group in groups
        }

        # Every measurement is keyed by its search radius rather than by the density
        # target it was chosen for: the real scans' radii are physical lengths with no
        # density target at all, so the targets are not unique within a group
        # (dataset, num_points, radius) -> realised mean neighbours per sphere
        self.realised_density: dict[tuple[str, int, float], float | None] = {}
        # (dataset, num_points, radius) -> the label this radius forms a column under
        self.column_label: dict[tuple[str, int, float], str] = {}
        # Column label -> sort key, so the selector lists columns in a sane order
        self.column_order: dict[str, tuple[int, float]] = {}
        # (dataset, num_points, radius, k, num_queries, competitor) -> seconds
        self.find_seconds: dict[tuple[str, int, float, int, int, str], float] = {}
        self.count_seconds: dict[tuple[str, int, float, int, str], float] = {}
        self.build_seconds: dict[tuple[str, int, float, str], float] = {}
        self.added_rss_mb: dict[tuple[str, int, float, str], float] = {}
        self.recall: dict[tuple[str, int, float, int, str], float] = {}
        self.index_agreement: dict[tuple[str, int, float, int, str], float] = {}
        self.max_distance_error: dict[tuple[str, int, float, int, str], float] = {}
        self.count_agreement: dict[tuple[str, int, float, str], float] = {}
        self.skipped: list[dict[str, Any]] = []

        self._collect()

    def _collect(self) -> None:
        """
        Index every record by the grid point it belongs to
        """
        for group in self.groups:
            dataset = group["dataset"]["label"]
            num_points = group["dataset"]["num_points"]
            for plan in group["radius_plans"]:
                self.realised_density[(dataset, num_points, plan["radius"])] = plan[
                    "realised_points_per_sphere"
                ]
                label = column_label(plan)
                self.column_label[(dataset, num_points, plan["radius"])] = label
                self.column_order[label] = column_sort_key(plan)

            for record in group["runs"]:
                radius = record["radius"]
                competitor = record["competitor"]
                workload = record["workload"]
                status = record.get("status")

                if status in ("skipped", "timeout", "failed"):
                    self.skipped.append(
                        {
                            "dataset": dataset,
                            "num_points": num_points,
                            "radius": radius,
                            "column": self.column_label.get(
                                (dataset, num_points, radius), f"r = {radius:.4g}"
                            ),
                            "competitor": competitor,
                            "workload": workload,
                            "num_neighbours": record.get("num_neighbours", 0),
                            "num_queries": record.get("num_queries", 0),
                            "status": status,
                            "detail": record.get("detail", ""),
                        }
                    )
                    continue

                if workload == "build":
                    self.build_seconds[(dataset, num_points, radius, competitor)] = record[
                        "build_seconds"
                    ]
                    if "added_rss_mb" in record:
                        self.added_rss_mb[(dataset, num_points, radius, competitor)] = record[
                            "added_rss_mb"
                        ]
                elif workload == "find":
                    self.find_seconds[
                        (
                            dataset,
                            num_points,
                            radius,
                            record["num_neighbours"],
                            record["num_queries"],
                            competitor,
                        )
                    ] = record["query_seconds"]
                elif workload == "count":
                    self.count_seconds[
                        (dataset, num_points, radius, record["num_queries"], competitor)
                    ] = record["query_seconds"]
                elif workload == "check":
                    correctness = record["correctness"]
                    key = (dataset, num_points, radius, record["num_neighbours"], competitor)
                    self.recall[key] = correctness["distance_recall"]
                    self.index_agreement[key] = correctness["index_agreement"]
                    self.max_distance_error[key] = correctness["max_distance_error"]
                elif workload == "check-count":
                    self.count_agreement[(dataset, num_points, radius, competitor)] = record[
                        "correctness"
                    ]["count_agreement"]

    def grid_points(self) -> list[tuple[str, int, float, int, int]]:
        """
        Every `(dataset, N, radius, k, Q)` the kNN workload was measured at

        Returns:
            The grid points, in a stable order
        """
        return sorted(
            {key[:5] for key in self.find_seconds},
            key=lambda point: (point[0], point[1], point[2], point[3], point[4]),
        )

    def fastest_exact(
        self, grid_point: tuple[str, int, float, int, int]
    ) -> tuple[str, float] | None:
        """
        The fastest exact method at one grid point

        Args:
            grid_point: `(dataset, N, radius, k, Q)`

        Returns:
            The competitor key and its query time, or None if nothing exact ran there
        """
        candidates = [
            (self.find_seconds[(*grid_point, competitor)], competitor)
            for competitor in self.exact_competitors
            if (*grid_point, competitor) in self.find_seconds
        ]
        if not candidates:
            return None
        seconds, competitor = min(candidates)
        return competitor, seconds

    def speed_up(
        self, grid_point: tuple[str, int, float, int, int], competitor: str
    ) -> float | None:
        """
        How many times faster than the reference a competitor was at a grid point

        Args:
            grid_point: `(dataset, N, radius, k, Q)`
            competitor: Competitor key

        Returns:
            The ratio, or None when either measurement is missing
        """
        reference = self.find_seconds.get((*grid_point, REFERENCE))
        measured = self.find_seconds.get((*grid_point, competitor))
        if not reference or not measured:
            return None
        return reference / measured

    def win_counts(self) -> dict[str, int]:
        """
        How many grid points each exact method is the fastest on

        Returns:
            Competitor key -> number of grid points won, descending
        """
        counts: dict[str, int] = defaultdict(int)
        for grid_point in self.grid_points():
            winner = self.fastest_exact(grid_point)
            if winner is not None:
                counts[winner[0]] += 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def chart_payload(self) -> dict[str, Any]:
        """
        The compact tables the charts are drawn from, ready to embed as JSON

        Rows are plain arrays rather than objects: at several thousand measurements
        that is the difference between a 200 kB and a 1 MB report

        Returns:
            Dict of metadata plus `find`, `count` and `recall` row tables
        """
        dataset_index = {label: number for number, label in enumerate(self.dataset_labels)}
        competitor_index = {key: number for number, key in enumerate(self.competitors)}
        columns = sorted(self.column_order, key=lambda label: self.column_order[label])
        column_index = {label: number for number, label in enumerate(columns)}

        def column_of(dataset: str, num_points: int, radius: float) -> int:
            return column_index[self.column_label[(dataset, num_points, radius)]]

        find_rows = [
            [
                dataset_index[dataset],
                num_points,
                column_of(dataset, num_points, radius),
                num_neighbours,
                num_queries,
                competitor_index[competitor],
                round(seconds, 6),
                round(self.build_seconds.get((dataset, num_points, radius, competitor), 0.0), 6),
            ]
            for (
                dataset,
                num_points,
                radius,
                num_neighbours,
                num_queries,
                competitor,
            ), seconds in sorted(self.find_seconds.items())
        ]
        count_rows = [
            [
                dataset_index[dataset],
                num_points,
                column_of(dataset, num_points, radius),
                num_queries,
                competitor_index[competitor],
                round(seconds, 6),
            ]
            for (dataset, num_points, radius, num_queries, competitor), seconds in sorted(
                self.count_seconds.items()
            )
        ]
        recall_rows = [
            [
                dataset_index[dataset],
                num_points,
                column_of(dataset, num_points, radius),
                num_neighbours,
                round(value, 4),
            ]
            for (
                dataset,
                num_points,
                radius,
                num_neighbours,
                competitor,
            ), value in sorted(self.recall.items())
            if not self.competitor_specs[competitor]["exact"]
        ]
        realised_rows = [
            [
                dataset_index[dataset],
                num_points,
                column_of(dataset, num_points, radius),
                round(value, 2) if value is not None else None,
                round(radius, 5),
            ]
            for (dataset, num_points, radius), value in sorted(self.realised_density.items())
        ]

        return {
            "datasets": self.dataset_labels,
            "columns": columns,
            "descriptions": [self.descriptions[label] for label in self.dataset_labels],
            "competitors": [
                {
                    "key": key,
                    "label": self.competitor_specs[key]["label"],
                    "exact": self.competitor_specs[key]["exact"],
                    "light": SERIES_COLOURS.get(key, ("#898781", "#898781"))[0],
                    "dark": SERIES_COLOURS.get(key, ("#898781", "#898781"))[1],
                }
                for key in self.competitors
            ],
            "reference": self.competitors.index(REFERENCE) if REFERENCE in self.competitors else -1,
            "find": find_rows,
            "count": count_rows,
            "recall": recall_rows,
            "realised": realised_rows,
            "ramp": list(SEQUENTIAL_RAMP),
        }


def render_summary_tiles(model: ReportModel) -> str:
    """
    The headline numbers, as stat tiles rather than a chart

    Args:
        model: The extracted results

    Returns:
        HTML for the tile row
    """
    grid_points = model.grid_points()
    win_counts = model.win_counts()
    total_wins = sum(win_counts.values())
    leader, leader_wins = next(iter(win_counts.items()), ("none", 0))
    leader_label = model.competitor_specs.get(leader, {}).get("label", leader)

    speed_ups = sorted(
        value
        for grid_point in grid_points
        if (value := model.speed_up(grid_point, leader)) is not None
    )
    median_speed_up = speed_ups[len(speed_ups) // 2] if speed_ups else float("nan")
    worst_speed_up = speed_ups[0] if speed_ups else float("nan")

    tiles = [
        ("Grid points measured", f"{len(grid_points)}", "dataset x N x radius x k x Q"),
        (
            "Fastest exact method",
            leader_label,
            f"{leader_wins} of {total_wins} grid points ({leader_wins / max(total_wins, 1):.0%})",
        ),
        (
            "Median speed-up vs scipy",
            f"{median_speed_up:.1f}x",
            f"{leader_label}; its worst grid point is {1 / worst_speed_up:.1f}x slower",
        ),
        (
            "Datasets",
            f"{len(model.dataset_labels)}",
            f"{len(model.groups)} (dataset, N) groups",
        ),
    ]
    cells = "\n".join(
        f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
        f'<div class="tile-value">{html.escape(value)}</div>'
        f'<div class="tile-note">{html.escape(note)}</div></div>'
        for label, value, note in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def render_headline_table(model: ReportModel) -> str:
    """
    Per dataset: the fastest exact method at small k, large k and on counting

    Args:
        model: The extracted results

    Returns:
        HTML table
    """
    rows: list[str] = []
    for dataset in model.dataset_labels:
        cells = [html.escape(dataset)]
        for num_neighbours in sorted({point[3] for point in model.grid_points()}):
            wins: dict[str, int] = defaultdict(int)
            speed_ups: dict[str, list[float]] = defaultdict(list)
            for grid_point in model.grid_points():
                if grid_point[0] != dataset or grid_point[3] != num_neighbours:
                    continue
                winner = model.fastest_exact(grid_point)
                if winner is None:
                    continue
                wins[winner[0]] += 1
                speed_up = model.speed_up(grid_point, winner[0])
                if speed_up is not None:
                    speed_ups[winner[0]].append(speed_up)
            if not wins:
                cells.append("-")
                continue
            best = max(wins.items(), key=lambda item: item[1])
            margins = sorted(speed_ups[best[0]])
            median = margins[len(margins) // 2] if margins else float("nan")
            label = model.competitor_specs[best[0]]["label"].replace("oxvox ", "")
            cells.append(
                f"{html.escape(label)}<span class='muted'> {best[1]}/{sum(wins.values())},"
                f" {median:.1f}x scipy</span>"
            )

        # Counting: the fastest method summed over every count measurement
        count_wins: dict[str, int] = defaultdict(int)
        count_speed_ups: dict[str, list[float]] = defaultdict(list)
        count_points = {
            key[:4] for key in model.count_seconds if key[0] == dataset
        }
        for point in count_points:
            candidates = [
                (model.count_seconds[(*point, competitor)], competitor)
                for competitor in model.exact_competitors
                if (*point, competitor) in model.count_seconds
            ]
            if not candidates:
                continue
            seconds, competitor = min(candidates)
            count_wins[competitor] += 1
            reference = model.count_seconds.get((*point, REFERENCE))
            if reference:
                count_speed_ups[competitor].append(reference / seconds)
        if count_wins:
            best_count = max(count_wins.items(), key=lambda item: item[1])
            margins = sorted(count_speed_ups[best_count[0]])
            median = margins[len(margins) // 2] if margins else float("nan")
            label = model.competitor_specs[best_count[0]]["label"].replace("oxvox ", "")
            cells.append(
                f"{html.escape(label)}<span class='muted'> {best_count[1]}/"
                f"{sum(count_wins.values())}, {median:.1f}x scipy</span>"
            )
        else:
            cells.append("-")

        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    header_cells = ["Dataset"] + [
        f"k = {num_neighbours}" for num_neighbours in sorted({point[3] for point in model.grid_points()})
    ] + ["count_neighbours"]
    header = "".join(f"<th>{html.escape(cell)}</th>" for cell in header_cells)
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def render_winner_table(model: ReportModel) -> str:
    """
    The fastest exact method at every grid point, with its margin over the runner-up

    Args:
        model: The extracted results

    Returns:
        HTML table, filterable by dataset through the select rendered above it
    """
    rows: list[str] = []
    for grid_point in model.grid_points():
        dataset, num_points, radius, num_neighbours, num_queries = grid_point
        candidates = sorted(
            (model.find_seconds[(*grid_point, competitor)], competitor)
            for competitor in model.exact_competitors
            if (*grid_point, competitor) in model.find_seconds
        )
        if not candidates:
            continue
        best_seconds, best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        speed_up = model.speed_up(grid_point, best)
        realised = model.realised_density.get((dataset, num_points, radius))
        rows.append(
            "<tr data-dataset=\"{dataset}\">"
            "<td>{dataset}</td><td>{points}</td><td>{density}</td><td>{k}</td><td>{queries}</td>"
            "<td>{winner}</td><td>{time}</td><td>{margin}</td><td>{speed_up}</td></tr>".format(
                dataset=html.escape(dataset),
                points=_format_count(num_points),
                density=(
                    f"{realised:.3g}"
                    if realised is not None
                    else html.escape(model.column_label[(dataset, num_points, radius)])
                ),
                k=num_neighbours,
                queries=_format_count(num_queries),
                winner=html.escape(
                    model.competitor_specs[best]["label"].replace("oxvox ", "")
                ),
                time=_format_seconds(best_seconds),
                margin=(f"{runner_up[0] / best_seconds:.2f}x" if runner_up else "-"),
                speed_up=(f"{speed_up:.2f}x" if speed_up else "-"),
            )
        )

    options = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)}</option>'
        for label in model.dataset_labels
    )
    return f"""
<div class="controls">
  <label>Dataset
    <select id="winner-filter"><option value="">all</option>{options}</select>
  </label>
</div>
<div class="table-scroll"><table id="winner-table">
<thead><tr><th>Dataset</th><th>N</th><th>Points per sphere</th><th>k</th><th>Queries</th>
<th>Fastest exact</th><th>Query time</th><th>Margin over 2nd</th><th>vs scipy</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
"""


def render_correctness_table(model: ReportModel) -> str:
    """
    One row per implementation: the worst agreement it showed anywhere in the grid

    Args:
        model: The extracted results

    Returns:
        HTML table
    """
    rows: list[str] = []
    for competitor in model.competitors:
        recalls = [
            value for key, value in model.recall.items() if key[4] == competitor
        ]
        agreements = [
            value for key, value in model.index_agreement.items() if key[4] == competitor
        ]
        errors = [
            value for key, value in model.max_distance_error.items() if key[4] == competitor
        ]
        count_agreements = [
            value for key, value in model.count_agreement.items() if key[3] == competitor
        ]
        if not recalls and not count_agreements:
            continue
        spec = model.competitor_specs[competitor]
        rows.append(
            "<tr><td>{label}</td><td>{exact}</td><td>{checks}</td><td>{recall}</td>"
            "<td>{agreement}</td><td>{error}</td><td>{counts}</td></tr>".format(
                label=html.escape(spec["label"]),
                exact="exact" if spec["exact"] else "approximate",
                checks=len(recalls),
                recall=f"{min(recalls):.4f}" if recalls else "-",
                agreement=f"{min(agreements):.4f}" if agreements else "-",
                error=f"{max(errors):.2e} m" if errors else "-",
                counts=f"{min(count_agreements):.4f}" if count_agreements else "-",
            )
        )
    return (
        '<div class="table-scroll"><table><thead><tr><th>Implementation</th><th>Claim</th>'
        "<th>Checks</th><th>Worst recall</th><th>Worst index agreement</th>"
        "<th>Largest distance error</th><th>Worst count agreement</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_correctness_note(model: ReportModel) -> str:
    """
    A sentence on the only disagreement the exact implementations show, from the data

    Args:
        model: The extracted results

    Returns:
        HTML paragraph, or an empty string if nothing disagreed
    """
    disagreements = [
        (key, value)
        for key, value in model.count_agreement.items()
        if value < 1.0 and model.competitor_specs[key[3]]["exact"]
    ]
    if not disagreements:
        return ""
    datasets = sorted({key[0] for key, _ in disagreements})
    radii = sorted({key[2] for key, _ in disagreements})
    worst = min(value for _, value in disagreements)
    return (
        f"<p>The exact backends reproduce scipy's neighbour counts on every check but "
        f"{len(disagreements)} of {len(model.count_agreement)}, and there they differ by "
        f"one neighbour in {1 - worst:.1%} of query points, only on "
        f"{', '.join(html.escape(dataset) for dataset in datasets)} at "
        f"{', '.join(f'{radius:g} m' for radius in radii)}. That is the radius boundary, "
        "not a bug: scipy counts a point at exactly the radius, oxvox does not "
        "(<code>d&sup2; &lt; r&sup2;</code>), and a real scan's quantised coordinates put "
        "points exactly there.</p>"
    )


def render_recall_note(model: ReportModel) -> str:
    """
    The approximate backend's recall, as numbers rather than only as a heatmap

    Args:
        model: The extracted results

    Returns:
        HTML paragraph
    """
    recalls = sorted(
        value
        for key, value in model.recall.items()
        if not model.competitor_specs[key[4]]["exact"]
    )
    counts = sorted(
        value
        for key, value in model.count_agreement.items()
        if not model.competitor_specs[key[3]]["exact"]
    )
    if not recalls:
        return ""
    above_99 = sum(1 for value in recalls if value >= 0.99) / len(recalls)
    note = (
        f"<p>Over {len(recalls)} checks its kNN recall has a median of "
        f"{recalls[len(recalls) // 2]:.4f}, a minimum of {recalls[0]:.4f}, and is at least "
        f"0.99 on {above_99:.0%} of them."
    )
    if counts:
        note += (
            f" Counting is the workload it cannot do: its exact-count agreement falls as low "
            f"as {counts[0]:.1%} of query points on the dense clouds, because a flood that "
            "stops at the k-th best neighbour has no reason to visit every point in the "
            "radius."
        )
    return note + "</p>"


def render_skipped_table(model: ReportModel) -> str:
    """
    The configurations that were not measured, and why

    Args:
        model: The extracted results

    Returns:
        HTML table, or a note when nothing was skipped
    """
    if not model.skipped:
        return "<p>Every configuration in the grid was measured.</p>"
    rows = "".join(
        "<tr><td>{dataset}</td><td>{points}</td><td>{column}</td><td>{competitor}</td>"
        "<td>{workload}</td><td>{k}</td><td>{queries}</td><td>{status}</td>"
        "<td>{detail}</td></tr>".format(
            dataset=html.escape(entry["dataset"]),
            points=_format_count(entry["num_points"]),
            column=html.escape(entry["column"]),
            competitor=html.escape(entry["competitor"]),
            workload=html.escape(entry["workload"]),
            k=entry["num_neighbours"] or "-",
            queries=_format_count(entry["num_queries"]) if entry["num_queries"] else "-",
            status=html.escape(entry["status"]),
            detail=html.escape(entry["detail"]),
        )
        for entry in model.skipped
    )
    return (
        f"<p>{len(model.skipped)} configurations were skipped, each because the harness "
        "projected it past a time or memory limit (see the run's options in the results "
        "JSON).</p>"
        '<div class="table-scroll"><table><thead><tr><th>Dataset</th><th>N</th>'
        "<th>Radius column</th><th>Implementation</th><th>Workload</th><th>k</th>"
        "<th>Queries</th><th>Status</th><th>Reason</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></div>"
    )


# The page: one HTML file, with the palette as CSS custom properties so light and dark
# swap in one place, and the chart code reading its colours back out of them
PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>oxvox neighbour search benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.0.1/plotly.min.js"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 16px 64px;
  background: var(--page);
  color: var(--text-primary);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; }
header { padding: 32px 0 8px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 40px 0 4px; }
h3 { font-size: 13px; margin: 0 0 4px; color: var(--text-secondary); font-weight: 600; }
p { margin: 8px 0; color: var(--text-secondary); max-width: 74ch; }
a { color: inherit; }
.meta { color: var(--muted); font-size: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;
         margin: 20px 0 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
        padding: 14px 16px; }
.tile-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
              color: var(--muted); }
.tile-value { font-size: 26px; margin: 4px 0 2px; color: var(--text-primary); }
.tile-note { font-size: 12px; color: var(--text-secondary); }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin: 16px 0 8px; }
.controls label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
                  color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
select { font: inherit; text-transform: none; letter-spacing: normal; color: var(--text-primary);
         background: var(--surface); border: 1px solid var(--border); border-radius: 7px;
         padding: 6px 8px; min-width: 130px; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 4px 0 10px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px;
               color: var(--text-secondary); }
.legend-swatch { width: 14px; height: 3px; border-radius: 2px; flex: none; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
         padding: 12px 12px 4px; }
.plot { width: 100%; height: 260px; }
.plot-wide { width: 100%; height: 360px; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid);
         white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.04em; }
.muted { color: var(--muted); }
.table-scroll { overflow-x: auto; max-height: 520px; overflow-y: auto;
                border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
                margin-top: 8px; }
footer { margin-top: 48px; color: var(--muted); font-size: 12px; }
.theme-toggle { position: absolute; top: 16px; right: 16px; }
</style>
</head>
<body>
<div class="theme-toggle">
  <label class="muted" style="font-size:11px">theme
    <select id="theme-select">
      <option value="system">system</option>
      <option value="light">light</option>
      <option value="dark">dark</option>
    </select>
  </label>
</div>
<main>
<header>
  <h1>oxvox neighbour search benchmark</h1>
  <p>__LEAD__</p>
  <p class="meta">__META__</p>
</header>

__TILES__

<h2>Headline: the fastest exact method per dataset family</h2>
<p>Each cell names the exact method that won the most grid points in that column, how
many of them it won, and the median factor by which it beat scipy's cKDTree there.</p>
__HEADLINE__

<h2>Query time against cloud size</h2>
<p>One panel per dataset, log-log. Pick the neighbour count, the density and the query
batch; switch the metric to see the same measurements as a speed-up over scipy, where
the dotted line at 1.0 is scipy itself. A missing point is a configuration the harness
skipped as projected too slow &mdash; the skip table at the bottom says which.</p>
<div class="controls">
  <label>Metric
    <select id="metric">
      <option value="time">query time (s)</option>
      <option value="speedup">speed-up vs scipy</option>
    </select>
  </label>
  <label>Neighbours (k)<select id="k"></select></label>
  <label>Radius column<select id="density"></select></label>
  <label>Query batch<select id="queries"></select></label>
</div>
<div class="legend" id="time-legend"></div>
<div class="panel-grid" id="time-grid"></div>

<h2>Where the index build stops paying for itself</h2>
<p>Build time plus query time against the number of queries answered, for one cloud.
A method with a cheaper build wins the left of the chart however fast its queries are;
the crossing point is the batch size above which the better query time takes over.</p>
<div class="controls">
  <label>Cloud<select id="bq-group"></select></label>
  <label>Radius column<select id="bq-density"></select></label>
  <label>Neighbours (k)<select id="bq-k"></select></label>
</div>
<div class="legend" id="build-legend"></div>
<div class="panel"><div id="build-plot" class="plot-wide"></div></div>

<h2>Counting neighbours</h2>
<p>The <code>count_neighbours</code> workload: how many search points lie within the
radius of each query, with no k bound at all. Open3D is absent because it exposes no
count query. Log-log, one panel per dataset.</p>
<div class="controls">
  <label>Radius column<select id="count-density"></select></label>
  <label>Query batch<select id="count-queries"></select></label>
</div>
<div class="legend" id="count-legend"></div>
<div class="panel-grid" id="count-grid"></div>

<h2>How much the approximate backend misses</h2>
<p>Recall of the <code>graph</code> backend against scipy's answers, on the 2048-query
correctness subsample: the fraction of the true neighbours it returned. It never
returns a point outside the radius, so what recall measures is omission, not error.</p>
__RECALL_NOTE__
<div class="panel"><div id="recall-plot" class="plot-wide"></div></div>

<h2>Correctness</h2>
<p>Every exact implementation has to reproduce scipy's distances. Index sets are only
compared where the distances are all distinct, since a tie lets any implementation
return either point.</p>
__CORRECTNESS_NOTE__
__CORRECTNESS__

<h2>Fastest exact method at every grid point</h2>
__WINNERS__

<h2>What was not measured</h2>
__SKIPPED__

<footer>
<p>Generated by <code>python -m benchmarks.report</code> from the JSON files in
<code>benchmarks/results/</code>. Charts need Plotly from a CDN; the tables above do
not.</p>
</footer>
</main>
<script>
const DATA = __DATA__;
</script>
<script>
__SCRIPT__
</script>
</body>
</html>
"""

# Chart code. Kept out of the template above so that the braces of JavaScript never
# have to be escaped for Python's string formatting
REPORT_SCRIPT = r"""
const FIND = { dataset: 0, n: 1, column: 2, k: 3, q: 4, competitor: 5, query: 6, build: 7 };
const COUNT = { dataset: 0, n: 1, column: 2, q: 3, competitor: 4, query: 5 };
const RECALL = { dataset: 0, n: 1, column: 2, k: 3, value: 4 };

const uniqueSorted = (values) => [...new Set(values)].sort((a, b) => a - b);
const formatCount = (value) =>
  value >= 1e6 ? `${value / 1e6}M` : value >= 1e3 ? `${value / 1e3}k` : `${value}`;

/* The realised mean neighbours per sphere, and the radius, per (dataset, N, column) */
const realisedLookup = new Map(
  DATA.realised.map((row) => [`${row[0]}|${row[1]}|${row[2]}`, { density: row[3], radius: row[4] }])
);
const realised = (dataset, n, column) => realisedLookup.get(`${dataset}|${n}|${column}`);

function isDark() {
  const stamped = document.documentElement.dataset.theme;
  if (stamped === "dark") return true;
  if (stamped === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function ink() {
  const style = getComputedStyle(document.body);
  return {
    text: style.getPropertyValue("--text-primary").trim(),
    secondary: style.getPropertyValue("--text-secondary").trim(),
    muted: style.getPropertyValue("--muted").trim(),
    grid: style.getPropertyValue("--grid").trim(),
    axis: style.getPropertyValue("--axis").trim(),
  };
}

function seriesColour(competitorIndex) {
  const competitor = DATA.competitors[competitorIndex];
  return isDark() ? competitor.dark : competitor.light;
}

function baseLayout(xTitle, yTitle, options = {}) {
  const colours = ink();
  const axis = {
    gridcolor: colours.grid,
    zeroline: false,
    linecolor: colours.axis,
    tickfont: { size: 10, color: colours.muted },
    titlefont: { size: 11, color: colours.muted },
    automargin: true,
  };
  return {
    margin: { l: 52, r: 20, t: 8, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: colours.secondary },
    hovermode: "x unified",
    hoverlabel: { font: { size: 11 } },
    showlegend: options.showlegend ?? false,
    legend: {
      orientation: "h",
      y: 1.12,
      font: { size: 11, color: colours.secondary },
    },
    xaxis: { ...axis, title: { text: xTitle }, type: "log" },
    yaxis: { ...axis, title: { text: yTitle }, type: "log" },
    ...options.layout,
  };
}

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/* One legend per section, in HTML: eight Plotly legends would not fit a small panel,
   and identity must never rest on colour alone */
function renderLegend(containerId, competitorIndices) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = competitorIndices
    .map(
      (index) =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${seriesColour(
          index
        )}"></span>${DATA.competitors[index].label}</span>`
    )
    .join("");
}

/* Ticks only at the measured values: a log axis otherwise labels every minor tick and
   the labels collide */
function logTicks(values) {
  return { tickvals: values, ticktext: values.map(formatCount) };
}

/* ---------- query time / speed-up small multiples ---------- */

function fillSelect(id, values, format, initial) {
  const select = document.getElementById(id);
  select.innerHTML = values
    .map((value) => `<option value="${value}">${format(value)}</option>`)
    .join("");
  select.value = initial !== undefined ? initial : values[values.length - 1];
}

function columnLabel(columnIndex) {
  return DATA.columns[columnIndex];
}

function renderTimeGrid() {
  const metric = document.getElementById("metric").value;
  const wantedK = Number(document.getElementById("k").value);
  const wantedColumn = Number(document.getElementById("density").value);
  const wantedQueries = document.getElementById("queries").value;
  const container = document.getElementById("time-grid");
  container.innerHTML = "";
  renderLegend(
    "time-legend",
    competitorsPresent(DATA.find, FIND.competitor).filter(
      (index) => metric !== "speedup" || index !== DATA.reference
    )
  );
  const pending = [];

  DATA.datasets.forEach((label, datasetIndex) => {
    const rows = DATA.find.filter(
      (row) =>
        row[FIND.dataset] === datasetIndex &&
        row[FIND.k] === wantedK &&
        row[FIND.column] === wantedColumn
    );
    if (!rows.length) return;

    /* Which query batch each cloud size is shown at: a fixed size, or the largest
       batch that any implementation managed there */
    const sizes = uniqueSorted(rows.map((row) => row[FIND.n]));
    const chosenQueries = new Map();
    sizes.forEach((n) => {
      const available = uniqueSorted(
        rows.filter((row) => row[FIND.n] === n).map((row) => row[FIND.q])
      );
      if (wantedQueries === "max") {
        chosenQueries.set(n, available[available.length - 1]);
      } else if (available.includes(Number(wantedQueries))) {
        chosenQueries.set(n, Number(wantedQueries));
      }
    });

    const traces = [];
    DATA.competitors.forEach((competitor, competitorIndex) => {
      if (metric === "speedup" && competitorIndex === DATA.reference) return;
      const x = [];
      const y = [];
      const text = [];
      sizes.forEach((n) => {
        const queries = chosenQueries.get(n);
        if (queries === undefined) return;
        const row = rows.find(
          (candidate) =>
            candidate[FIND.n] === n &&
            candidate[FIND.q] === queries &&
            candidate[FIND.competitor] === competitorIndex
        );
        if (!row) return;
        let value = row[FIND.query];
        if (metric === "speedup") {
          const reference = rows.find(
            (candidate) =>
              candidate[FIND.n] === n &&
              candidate[FIND.q] === queries &&
              candidate[FIND.competitor] === DATA.reference
          );
          if (!reference) return;
          value = reference[FIND.query] / row[FIND.query];
        }
        x.push(n);
        y.push(value);
        text.push(`${formatCount(queries)} queries`);
      });
      if (!x.length) return;
      traces.push({
        x,
        y,
        text,
        name: competitor.label,
        type: "scatter",
        mode: "lines+markers",
        line: { color: seriesColour(competitorIndex), width: 2 },
        marker: { size: 8, color: seriesColour(competitorIndex) },
        hovertemplate:
          metric === "speedup"
            ? "%{fullData.name}: %{y:.2f}x scipy<extra></extra>"
            : "%{fullData.name}: %{y:.4g} s (%{text})<extra></extra>",
      });
    });
    if (!traces.length) return;

    if (metric === "speedup") {
      traces.push({
        x: sizes,
        y: sizes.map(() => 1),
        name: "scipy cKDTree",
        type: "scatter",
        mode: "lines",
        line: { color: ink().muted, width: 2, dash: "dot" },
        hoverinfo: "skip",
      });
    }

    const entry = realised(datasetIndex, sizes[0], wantedColumn);
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML =
      `<h3>${label}</h3>` +
      `<div class="meta" style="font-size:11px">${
        entry && entry.density != null
          ? `${Number(entry.density).toPrecision(3)} points per sphere at N=${formatCount(
              sizes[0]
            )}, r = ${entry.radius} m`
          : ""
      }</div>` +
      `<div class="plot" id="time-plot-${datasetIndex}"></div>`;
    container.appendChild(panel);
    pending.push({ id: `time-plot-${datasetIndex}`, traces, sizes });
  });

  // Plot only once every panel is in the DOM: a plot drawn while the grid is still
  // growing measures a container width that the next panel then takes away from it
  pending.forEach(({ id, traces, sizes }) =>
    Plotly.newPlot(
      id,
      traces,
      baseLayout(
        "search points (N)",
        metric === "speedup" ? "times faster than scipy" : "query time (s)",
        {
          layout: {
            xaxis: {
              ...baseLayout("", "").xaxis,
              title: { text: "search points (N)" },
              ...logTicks(sizes),
            },
          },
        }
      ),
      PLOT_CONFIG
    )
  );
}

/* ---------- build + query against query count ---------- */

function groupKey(datasetIndex, n) {
  return `${datasetIndex}:${n}`;
}

function renderBuildChart() {
  const [datasetIndex, n] = document.getElementById("bq-group").value.split(":").map(Number);
  const column = Number(document.getElementById("bq-density").value);
  const wantedK = Number(document.getElementById("bq-k").value);
  const rows = DATA.find.filter(
    (row) =>
      row[FIND.dataset] === datasetIndex &&
      row[FIND.n] === n &&
      row[FIND.column] === column &&
      row[FIND.k] === wantedK
  );
  const queries = uniqueSorted(rows.map((row) => row[FIND.q]));
  renderLegend("build-legend", competitorsPresent(rows, FIND.competitor));
  const traces = [];
  DATA.competitors.forEach((competitor, competitorIndex) => {
    const own = rows.filter((row) => row[FIND.competitor] === competitorIndex);
    if (!own.length) return;
    traces.push({
      x: own.map((row) => row[FIND.q]),
      y: own.map((row) => row[FIND.build] + row[FIND.query]),
      customdata: own.map((row) => [row[FIND.build], row[FIND.query]]),
      name: competitor.label,
      type: "scatter",
      mode: "lines+markers",
      line: { color: seriesColour(competitorIndex), width: 2 },
      marker: { size: 8, color: seriesColour(competitorIndex) },
      hovertemplate:
        "%{fullData.name}: %{y:.4g} s total" +
        " (build %{customdata[0]:.4g} s + query %{customdata[1]:.4g} s)<extra></extra>",
    });
  });
  Plotly.newPlot(
    "build-plot",
    traces,
    baseLayout("queries answered", "build + query time (s)", {
      layout: {
        xaxis: {
          ...baseLayout("", "").xaxis,
          title: { text: "queries answered" },
          ...logTicks(queries),
        },
      },
    }),
    PLOT_CONFIG
  );
  if (!traces.length) {
    document.getElementById("build-plot").innerHTML =
      '<p class="muted">nothing measured for this combination</p>';
  }
}

/* ---------- count workload ---------- */

function renderCountGrid() {
  const column = Number(document.getElementById("count-density").value);
  const wantedQueries = document.getElementById("count-queries").value;
  const container = document.getElementById("count-grid");
  container.innerHTML = "";
  renderLegend("count-legend", competitorsPresent(DATA.count, COUNT.competitor));
  const pending = [];

  DATA.datasets.forEach((label, datasetIndex) => {
    const rows = DATA.count.filter(
      (row) => row[COUNT.dataset] === datasetIndex && row[COUNT.column] === column
    );
    if (!rows.length) return;
    const sizes = uniqueSorted(rows.map((row) => row[COUNT.n]));
    const chosenQueries = new Map();
    sizes.forEach((n) => {
      const available = uniqueSorted(
        rows.filter((row) => row[COUNT.n] === n).map((row) => row[COUNT.q])
      );
      if (wantedQueries === "max") {
        chosenQueries.set(n, available[available.length - 1]);
      } else if (available.includes(Number(wantedQueries))) {
        chosenQueries.set(n, Number(wantedQueries));
      }
    });

    const traces = [];
    DATA.competitors.forEach((competitor, competitorIndex) => {
      const x = [];
      const y = [];
      const text = [];
      sizes.forEach((n) => {
        const queries = chosenQueries.get(n);
        if (queries === undefined) return;
        const row = rows.find(
          (candidate) =>
            candidate[COUNT.n] === n &&
            candidate[COUNT.q] === queries &&
            candidate[COUNT.competitor] === competitorIndex
        );
        if (!row) return;
        x.push(n);
        y.push(row[COUNT.query]);
        text.push(`${formatCount(queries)} queries`);
      });
      if (!x.length) return;
      traces.push({
        x,
        y,
        text,
        name: competitor.label,
        type: "scatter",
        mode: "lines+markers",
        line: { color: seriesColour(competitorIndex), width: 2 },
        marker: { size: 8, color: seriesColour(competitorIndex) },
        hovertemplate: "%{fullData.name}: %{y:.4g} s (%{text})<extra></extra>",
      });
    });
    if (!traces.length) return;
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h3>${label}</h3><div class="plot" id="count-plot-${datasetIndex}"></div>`;
    container.appendChild(panel);
    pending.push({ id: `count-plot-${datasetIndex}`, traces, sizes });
  });

  pending.forEach(({ id, traces, sizes }) =>
    Plotly.newPlot(
      id,
      traces,
      baseLayout("search points (N)", "count time (s)", {
        layout: {
          xaxis: {
            ...baseLayout("", "").xaxis,
            title: { text: "search points (N)" },
            ...logTicks(sizes),
          },
        },
      }),
      PLOT_CONFIG
    )
  );
}

/* ---------- recall of the approximate backend ---------- */

function renderRecall() {
  const columns = [];
  const columnKeys = [];
  uniqueSorted(DATA.recall.map((row) => row[RECALL.column])).forEach((columnIndex) => {
    uniqueSorted(DATA.recall.map((row) => row[RECALL.k])).forEach((k) => {
      columnKeys.push(`${columnIndex}|${k}`);
      columns.push(`${columnLabel(columnIndex)}, k=${k}`);
    });
  });
  const rowKeys = [];
  const rowLabels = [];
  DATA.datasets.forEach((label, datasetIndex) => {
    uniqueSorted(
      DATA.recall.filter((row) => row[RECALL.dataset] === datasetIndex).map((row) => row[RECALL.n])
    ).forEach((n) => {
      rowKeys.push(`${datasetIndex}|${n}`);
      rowLabels.push(`${label} @ ${formatCount(n)}`);
    });
  });

  const lookup = new Map(
    DATA.recall.map((row) => [
      `${row[RECALL.dataset]}|${row[RECALL.n]}|${row[RECALL.column]}|${row[RECALL.k]}`,
      row[RECALL.value],
    ])
  );
  const z = rowKeys.map((rowKey) =>
    columnKeys.map((columnKey) => {
      const [datasetIndex, n] = rowKey.split("|");
      const [columnIndex, k] = columnKey.split("|");
      const value = lookup.get(`${datasetIndex}|${n}|${columnIndex}|${k}`);
      return value === undefined ? null : value;
    })
  );
  if (!rowKeys.length) {
    document.getElementById("recall-plot").innerHTML =
      '<p class="muted">the approximate backend was not measured in this run</p>';
    return;
  }
  const ramp = DATA.ramp.map((colour, index) => [index / (DATA.ramp.length - 1), colour]);
  Plotly.newPlot(
    "recall-plot",
    [
      {
        z,
        x: columns,
        y: rowLabels,
        type: "heatmap",
        colorscale: ramp,
        zmin: Math.min(0.9, ...z.flat().filter((value) => value !== null)),
        zmax: 1,
        xgap: 2,
        ygap: 2,
        colorbar: { title: { text: "recall", font: { size: 11 } }, thickness: 10, len: 0.8 },
        hovertemplate: "%{y}<br>%{x}<br>recall %{z:.3f}<extra></extra>",
      },
    ],
    baseLayout("", "", {
      layout: {
        xaxis: { type: "category", tickfont: { size: 10 }, automargin: true },
        yaxis: { type: "category", tickfont: { size: 10 }, automargin: true },
        margin: { l: 150, r: 10, t: 8, b: 90 },
      },
    }),
    PLOT_CONFIG
  );
}

/* ---------- wiring ---------- */

function renderEverything() {
  renderTimeGrid();
  renderBuildChart();
  renderCountGrid();
  renderRecall();
}

/* Which competitors appear in a section, in the fixed slot order */
function competitorsPresent(rows, columnIndex) {
  const present = new Set(rows.map((row) => row[columnIndex]));
  return DATA.competitors.map((_, index) => index).filter((index) => present.has(index));
}

function setUpControls() {
  const kValues = uniqueSorted(DATA.find.map((row) => row[FIND.k]));
  const columns = uniqueSorted(DATA.find.map((row) => row[FIND.column]));
  const queryValues = uniqueSorted(DATA.find.map((row) => row[FIND.q]));

  fillSelect("k", kValues, (value) => `k = ${value}`, kValues.includes(8) ? 8 : kValues[0]);
  fillSelect("density", columns, columnLabel, columns.includes(1) ? 1 : columns[0]);
  const queriesSelect = document.getElementById("queries");
  queriesSelect.innerHTML =
    '<option value="max">largest measured</option>' +
    queryValues.map((value) => `<option value="${value}">${formatCount(value)}</option>`).join("");

  const groups = [];
  DATA.datasets.forEach((label, datasetIndex) => {
    uniqueSorted(
      DATA.find.filter((row) => row[FIND.dataset] === datasetIndex).map((row) => row[FIND.n])
    ).forEach((n) => groups.push([groupKey(datasetIndex, n), `${label} @ ${formatCount(n)}`]));
  });
  const groupSelect = document.getElementById("bq-group");
  groupSelect.innerHTML = groups
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  const preferred = groups.find(([, label]) => label.startsWith("uniform @ 1M"));
  groupSelect.value = preferred ? preferred[0] : groups[0][0];

  fillSelect("bq-density", columns, columnLabel, columns.includes(1) ? 1 : columns[0]);
  fillSelect("bq-k", kValues, (value) => `k = ${value}`, kValues.includes(8) ? 8 : kValues[0]);

  const countColumns = uniqueSorted(DATA.count.map((row) => row[COUNT.column]));
  fillSelect(
    "count-density",
    countColumns,
    columnLabel,
    countColumns.includes(1) ? 1 : countColumns[0]
  );
  const countQueries = uniqueSorted(DATA.count.map((row) => row[COUNT.q]));
  document.getElementById("count-queries").innerHTML =
    '<option value="max">largest measured</option>' +
    countQueries.map((value) => `<option value="${value}">${formatCount(value)}</option>`).join("");

  ["metric", "k", "density", "queries"].forEach((id) =>
    document.getElementById(id).addEventListener("change", renderTimeGrid)
  );
  ["bq-group", "bq-density", "bq-k"].forEach((id) =>
    document.getElementById(id).addEventListener("change", renderBuildChart)
  );
  ["count-density", "count-queries"].forEach((id) =>
    document.getElementById(id).addEventListener("change", renderCountGrid)
  );

  const themeSelect = document.getElementById("theme-select");
  themeSelect.addEventListener("change", () => {
    if (themeSelect.value === "system") {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = themeSelect.value;
    }
    renderEverything();
  });
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => renderEverything());

  const winnerFilter = document.getElementById("winner-filter");
  winnerFilter.addEventListener("change", () => {
    const wanted = winnerFilter.value;
    document.querySelectorAll("#winner-table tbody tr").forEach((row) => {
      row.hidden = Boolean(wanted) && row.dataset.dataset !== wanted;
    });
  });
}

if (typeof Plotly === "undefined") {
  document.querySelectorAll(".panel-grid, #build-plot, #recall-plot").forEach((node) => {
    node.innerHTML =
      '<p class="muted">charts need Plotly, which could not be loaded; the tables below ' +
      "carry the same conclusions</p>";
  });
  const winnerFilter = document.getElementById("winner-filter");
  if (winnerFilter) winnerFilter.disabled = true;
} else {
  setUpControls();
  renderEverything();
}
"""


def build_html(model: ReportModel) -> str:
    """
    Assemble the whole report page

    Args:
        model: The extracted results

    Returns:
        The HTML document
    """
    meta = model.meta
    versions = ", ".join(
        f"{name} {version}"
        for name, version in meta["library_versions"].items()
        if version != "not installed"
    )
    meta_line = (
        f"{meta['date']} &middot; {meta['cpu_count']} cores &middot; "
        f"oxvox {meta['oxvox_version']} at commit {meta['oxvox_commit']} "
        f"(methods: {', '.join(meta['oxvox_methods'])}) &middot; {versions} &middot; "
        f"Python {meta['python']} on {html.escape(meta['platform'])}"
    )
    lead = (
        "Every backend oxvox ships, against scipy's cKDTree and Open3D's hybrid search, "
        "over five synthetic pointcloud families and two real laser scans. Build time and "
        "query time are measured separately (warm-up, then the median of three), each "
        "index is built and queried in its own worker process, and every exact "
        "implementation's answers are checked against scipy's."
    )

    replacements = {
        "__LEAD__": lead,
        "__META__": meta_line,
        "__TILES__": render_summary_tiles(model),
        "__HEADLINE__": render_headline_table(model),
        "__CORRECTNESS__": render_correctness_table(model),
        "__CORRECTNESS_NOTE__": render_correctness_note(model),
        "__RECALL_NOTE__": render_recall_note(model),
        "__WINNERS__": render_winner_table(model),
        "__SKIPPED__": render_skipped_table(model),
        "__DATA__": json.dumps(model.chart_payload(), separators=(",", ":")),
        "__SCRIPT__": REPORT_SCRIPT,
    }
    page = PAGE_TEMPLATE
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


def main(argv: list[str] | None = None) -> int:
    """
    Build the report from the results directory

    Args:
        argv: Command-line arguments, defaulting to `sys.argv[1:]`

    Returns:
        Process exit status
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory holding the results JSON (default: benchmarks/results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the report (default: <results-dir>/report.html)",
    )
    arguments = parser.parse_args(argv)

    model = ReportModel(load_results(arguments.results_dir))
    output = arguments.output or arguments.results_dir / "report.html"
    output.write_text(build_html(model))
    logger.info(
        "wrote %s from %d groups, %d kNN measurements",
        output,
        len(model.groups),
        len(model.find_seconds),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
