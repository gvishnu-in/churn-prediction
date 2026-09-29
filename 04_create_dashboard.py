"""Create a standalone interactive HTML dashboard from the cleaned churn data."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "telco_churn_clean.csv"
OUTPUT_PATH = PROJECT_DIR / "dashboard" / "churn_dashboard.html"

REQUIRED_COLUMNS = {
    "Churn",
    "Contract",
    "tenure_group",
    "InternetService",
    "MonthlyCharges",
    "PaymentMethod",
    "num_services",
}

COLORS = {
    "blue": "#3B82F6",
    "teal": "#14B8A6",
    "red": "#EF4444",
    "navy": "#1E3A5F",
    "muted": "#64748B",
}


def churn_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column, observed=False)
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )


def add_churn_bar(
    figure: go.Figure,
    data: pd.DataFrame,
    column: str,
    row: int,
    col: int,
    title: str,
) -> None:
    summary = churn_summary(data, column)
    figure.add_trace(
        go.Bar(
            x=summary[column].astype(str),
            y=summary["churn_rate"] * 100,
            customdata=summary["customers"],
            marker_color=COLORS["blue"],
            hovertemplate=(
                "%{x}<br>Churn rate: %{y:.1f}%"
                "<br>Customers: %{customdata:,}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    figure.update_yaxes(title_text="Churn rate (%)", ticksuffix="%", row=row, col=col)
    figure.update_xaxes(title_text=title, tickangle=-18, row=row, col=col)


def build_dashboard(data: pd.DataFrame) -> str:
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if data.empty:
        raise ValueError("The cleaned dataset contains no customer rows.")

    churn_values = set(data["Churn"].dropna().unique())
    if not churn_values.issubset({0, 1}) or data["Churn"].isna().any():
        raise ValueError("The Churn column must contain only 0 and 1 values.")

    churn_rate = data["Churn"].mean() * 100
    churned = int(data["Churn"].sum())
    average_charge = data["MonthlyCharges"].mean()

    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Churn by contract",
            "Churn by customer tenure",
            "Churn by internet service",
            "Monthly charges by churn status",
            "Churn by payment method",
            "Churn by number of services",
        ),
        horizontal_spacing=0.11,
        vertical_spacing=0.22,
    )

    add_churn_bar(figure, data, "Contract", 1, 1, "Contract")

    tenure_order = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"]
    tenure = churn_summary(data, "tenure_group")
    tenure["tenure_group"] = tenure["tenure_group"].astype(str)
    tenure["_order"] = tenure["tenure_group"].map(
        {label: index for index, label in enumerate(tenure_order)}
    )
    tenure = tenure.sort_values("_order")
    figure.add_trace(
        go.Bar(
            x=tenure["tenure_group"],
            y=tenure["churn_rate"] * 100,
            customdata=tenure["customers"],
            marker_color=COLORS["teal"],
            hovertemplate=(
                "%{x}<br>Churn rate: %{y:.1f}%"
                "<br>Customers: %{customdata:,}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="Churn rate (%)", ticksuffix="%", row=1, col=2)
    figure.update_xaxes(title_text="Customer tenure", row=1, col=2)

    add_churn_bar(figure, data, "InternetService", 1, 3, "Internet service")

    for churn_value, label, color in (
        (0, "Retained", COLORS["blue"]),
        (1, "Churned", COLORS["red"]),
    ):
        charges = data.loc[data["Churn"] == churn_value, "MonthlyCharges"]
        figure.add_trace(
            go.Box(
                y=charges,
                name=label,
                marker_color=color,
                boxpoints=False,
                hovertemplate=f"{label}<br>Monthly charge: $%{{y:.2f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="Monthly charge ($)", row=2, col=1)
    figure.update_xaxes(title_text="Customer status", row=2, col=1)

    add_churn_bar(figure, data, "PaymentMethod", 2, 2, "Payment method")
    add_churn_bar(figure, data, "num_services", 2, 3, "Number of services")

    figure.update_layout(
        template="plotly_white",
        height=850,
        barmode="group",
        margin={"l": 65, "r": 35, "t": 70, "b": 65},
        font={"family": "Arial, sans-serif", "color": COLORS["navy"]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={"bgcolor": "white"},
    )

    chart_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Telco Customer Churn Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, sans-serif;
      background: #f1f5f9;
      color: {COLORS["navy"]};
    }}
    body {{ margin: 0; padding: 28px; }}
    main {{ max-width: 1500px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .subtitle {{ margin: 0 0 22px; color: {COLORS["muted"]}; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .kpi {{
      background: white;
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: 0 2px 10px #0f172a12;
    }}
    .kpi-label {{ color: {COLORS["muted"]}; font-size: 14px; }}
    .kpi-value {{ margin-top: 8px; font-size: 28px; font-weight: 700; }}
    .chart {{
      background: white;
      border-radius: 12px;
      padding: 8px;
      box-shadow: 0 2px 10px #0f172a12;
    }}
    .note {{ color: {COLORS["muted"]}; font-size: 13px; margin: 14px 2px; }}
    @media (max-width: 700px) {{
      body {{ padding: 14px; }}
      .kpis {{ grid-template-columns: 1fr; gap: 10px; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Telco Customer Churn</h1>
    <p class="subtitle">Interactive overview of churn patterns across {len(data):,} customers</p>
    <section class="kpis" aria-label="Key metrics">
      <article class="kpi">
        <div class="kpi-label">Overall churn rate</div>
        <div class="kpi-value">{churn_rate:.1f}%</div>
      </article>
      <article class="kpi">
        <div class="kpi-label">Customers churned</div>
        <div class="kpi-value">{churned:,}</div>
      </article>
      <article class="kpi">
        <div class="kpi-label">Average monthly charge</div>
        <div class="kpi-value">${average_charge:.2f}</div>
      </article>
    </section>
    <section class="chart" aria-label="Interactive churn charts">
      {chart_html}
    </section>
    <p class="note">Hover over charts for details. Use the Plotly toolbar to zoom, pan, or download a chart image. These are historical customer patterns, not individual churn predictions.</p>
  </main>
</body>
</html>
"""


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    dashboard_html = build_dashboard(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(dashboard_html, encoding="utf-8")
    print(f"Dashboard created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
