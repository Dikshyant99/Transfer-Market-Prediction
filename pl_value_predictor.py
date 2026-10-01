"""
Premier League market value predictor (linear regression) with player search
and seaborn/matplotlib charts.

Data sources
- football-data.org  -> player stats (goals, assists, matches played), age, position
- market_values.csv  -> market values from Transfermarkt (name, market_value_eur)

Usage
    Jupyter:   %run pl_value_predictor.py     (type names when prompted)
               then predict_player("Haaland") or predict_custom(...)
    Terminal:  python pl_value_predictor.py
"""
import difflib
import os
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

TOKEN = os.environ["FOOTBALL_DATA_TOKEN"]
SEASON = 2025                      # 2025/26 season = "last season"
SEASON_END = pd.Timestamp("2026-05-31")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": TOKEN}
BIG_SIX = (
    "arsenal", "chelsea", "liverpool",
    "manchester city", "manchester united", "tottenham",
)

# ---- Look & feel ------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
C_PRED, C_ACTUAL, C_GREY = "#3B82F6", "#F59E0B", "#CBD5E1"
TICKS_M = [1, 2, 5, 10, 20, 50, 100, 200]


def eur_m_axis(ax, which="both"):
    """Log axis labelled in €m."""
    f = FuncFormatter(lambda v, _: f"€{v:g}m")
    for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        if which in ("both", name):
            getattr(ax, f"set_{name}scale")("log")
            getattr(ax, f"set_{name}ticks")(TICKS_M)
            axis.set_major_formatter(f)
            axis.set_minor_formatter(NullFormatter())


# ---- Data -------------------------------------------------------------------
def normalize(name: str) -> str:
    """Lowercase and strip accents so names match across sources."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(name.lower().split())


def fetch_player_stats() -> pd.DataFrame:
    r = requests.get(
        f"{BASE}/competitions/PL/scorers",
        headers=HEADERS,
        params={"season": SEASON, "limit": 500},
        timeout=30,
    )
    r.raise_for_status()
    rows = []
    for s in r.json()["scorers"]:
        p = s["player"]
        rows.append(
            {
                "name": p["name"],
                "position": p.get("position") or p.get("section") or "Unknown",
                "dob": p.get("dateOfBirth"),
                "team": s["team"]["name"],
                "matches": s.get("playedMatches"),
                "goals": s.get("goals") or 0,
                "assists": s.get("assists") or 0,
            }
        )
    df = pd.DataFrame(rows)
    df["age"] = (SEASON_END - pd.to_datetime(df["dob"])).dt.days / 365.25
    return df


def build_dataset() -> pd.DataFrame:
    stats = fetch_player_stats()
    values = pd.read_csv("market_values.csv")
    stats["key"] = stats["name"].map(normalize)
    values["key"] = values["name"].map(normalize)
    df = stats.merge(values[["key", "market_value_eur"]], on="key", how="inner")
    df = df.dropna(subset=["age", "matches", "market_value_eur"])
    df = df.drop_duplicates(subset="key").reset_index(drop=True)
    print(f"Matched {len(df)} of {len(stats)} players with a market value")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["goals_per_match"] = d["goals"] / d["matches"].clip(lower=1)
    d["assists_per_match"] = d["assists"] / d["matches"].clip(lower=1)
    d["age_sq"] = d["age"] ** 2  # value rises, peaks mid-20s, then falls
    d["big_six"] = d["team"].map(normalize).str.startswith(BIG_SIX).astype(int)
    return pd.get_dummies(
        d[
            [
                "goals", "assists", "matches",
                "goals_per_match", "assists_per_match",
                "age", "age_sq", "big_six", "position",
            ]
        ],
        columns=["position"],
        drop_first=True,
    )


# ---- Training + overview charts --------------------------------------------
def train(df: pd.DataFrame, show_plot: bool = True):
    X = make_features(df)
    y = np.log1p(df["market_value_eur"])  # values are right-skewed -> model log

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pred))
    print(f"R²  : {r2:.3f}")
    print(f"MAE : €{mae:,.0f}")

    coefs = pd.Series(model.coef_, index=X.columns).sort_values()
    plot_overview(df.loc[X_test.index], np.expm1(pred), r2, mae, coefs, show_plot)
    return model, X


def plot_overview(test_df, pred_eur, r2, mae, coefs, show_plot=True):
    """Held-out accuracy scatter + feature importance."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    d = test_df.assign(pred_m=pred_eur / 1e6, actual_m=test_df["market_value_eur"] / 1e6)

    sns.scatterplot(
        data=d, x="actual_m", y="pred_m", hue="position",
        s=70, alpha=0.85, edgecolor="white", linewidth=0.6, palette="Set2", ax=ax1,
    )
    ax1.plot([1, 250], [1, 250], "--", color="#94A3B8", lw=1.2, label="Perfect prediction")
    for _, r in d.nlargest(5, "actual_m").iterrows():
        ax1.annotate(r["name"].split()[-1], (r["actual_m"], r["pred_m"]),
                     xytext=(6, -10), textcoords="offset points", fontsize=9, color="#334155")
    eur_m_axis(ax1)
    ax1.set_xlim(1, 250); ax1.set_ylim(1, 250)
    ax1.set_xlabel("Actual market value"); ax1.set_ylabel("Predicted market value")
    ax1.set_title("Held-out players: actual vs predicted", loc="left", fontweight="bold")
    ax1.text(0.03, 0.95, f"R² = {r2:.2f}\nMAE = €{mae / 1e6:.1f}m", transform=ax1.transAxes,
             va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec=C_GREY))
    ax1.legend(title="Position", fontsize=8, title_fontsize=9, loc="lower right")

    colors = np.where(coefs >= 0, C_PRED, C_ACTUAL)
    ax2.barh(coefs.index, coefs.values, color=colors)
    ax2.axvline(0, color="#64748B", lw=0.8)
    ax2.set_title("What drives value (log scale coefficients)", loc="left", fontweight="bold")
    ax2.set_xlabel("Effect on log(value)")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Premier League market value model", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("model_overview.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()


def fmt(eur: float) -> str:
    return f"€{eur / 1e6:,.1f}m"


# ---- Player search ----------------------------------------------------------
DF = MODEL = X_ALL = None  # filled in by init()


def plot_player(p, pred_eur):
    """Card for one player: predicted vs actual bars + position on the full scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    pred_m, actual_m = pred_eur / 1e6, p["market_value_eur"] / 1e6

    bars = ax1.barh(["Transfermarkt", "Model"], [actual_m, pred_m], color=[C_ACTUAL, C_PRED], height=0.55)
    for b, v in zip(bars, [actual_m, pred_m]):
        ax1.text(v + max(actual_m, pred_m) * 0.02, b.get_y() + b.get_height() / 2,
                 f"€{v:,.1f}m", va="center", fontweight="bold")
    ax1.set_xlim(0, max(actual_m, pred_m) * 1.25)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€{v:g}m"))
    ax1.grid(axis="y", visible=False)
    diff = (pred_m - actual_m) / actual_m * 100
    ax1.set_title(f"Model is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} the market value",
                  loc="left", fontsize=11)

    ax2.scatter(DF["market_value_eur"] / 1e6, DF["predicted_eur"] / 1e6, s=28, color=C_GREY, alpha=0.8)
    ax2.plot([1, 250], [1, 250], "--", color="#94A3B8", lw=1)
    ax2.scatter([actual_m], [pred_m], s=170, color="#EF4444", edgecolor="white", linewidth=1.5, zorder=5)
    ax2.annotate(p["name"], (actual_m, pred_m), xytext=(10, -14), textcoords="offset points",
                 fontweight="bold", color="#EF4444")
    eur_m_axis(ax2)
    ax2.set_xlim(1, 250); ax2.set_ylim(1, 250)
    ax2.set_xlabel("Transfermarkt value"); ax2.set_ylabel("Model value")
    ax2.set_title("Where they sit among all players", loc="left", fontsize=11)

    fig.suptitle(
        f"{p['name']}  ·  {p['team']}  ·  {p['position']}, age {p['age']:.0f}\n"
        f"{int(p['goals'])} goals · {int(p['assists'])} assists · {int(p['matches'])} matches",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    plt.show()


def _closest(q):
    """Typo-tolerant fallback: match query against individual name parts."""
    tokens = sorted({t for k in DF["key"] for t in k.split()})
    close = difflib.get_close_matches(q, tokens, n=2, cutoff=0.7)
    if not close:
        return DF.iloc[0:0]
    return DF[DF["key"].apply(lambda k: any(t in close for t in k.split()))]


def predict_player(query: str):
    """Search by (part of a) name; prints and plots the predicted market value."""
    q = normalize(query)
    hits = DF[DF["key"].str.contains(q, regex=False)]
    if hits.empty:
        hits = _closest(q)
        if hits.empty:
            print(f"No player found for '{query}'. Try just the surname.")
            return
        print(f"No exact match for '{query}' - showing closest:")
    for _, p in hits.head(3).iterrows():
        pred = p["predicted_eur"]
        print(f"\n{p['name']} ({p['team']}, {p['position']}, age {p['age']:.0f})")
        print(f"  Predicted value:     {fmt(pred)}")
        print(f"  Transfermarkt value: {fmt(p['market_value_eur'])}")
        plot_player(p, pred)


def predict_custom(goals, assists, matches, age, position, big_six=0):
    """Predict for a made-up player, e.g. predict_custom(15, 8, 34, 24, 'Offence', 1)."""
    row = pd.DataFrame(0.0, index=[0], columns=X_ALL.columns)
    row["goals"], row["assists"], row["matches"] = goals, assists, matches
    row["goals_per_match"] = goals / max(matches, 1)
    row["assists_per_match"] = assists / max(matches, 1)
    row["age"], row["age_sq"], row["big_six"] = age, age ** 2, big_six
    col = f"position_{position}"
    if col in row.columns:
        row[col] = 1
    print(f"Predicted value: {fmt(np.expm1(MODEL.predict(row)[0]))}")
    print("Positions in data:", sorted(DF["position"].unique()))


def init():
    global DF, MODEL, X_ALL
    DF = build_dataset()
    MODEL, X_ALL = train(DF)
    DF["predicted_eur"] = np.expm1(MODEL.predict(X_ALL))


if __name__ == "__main__":
    init()
    print("\nType a player name (e.g. Haaland), or press Enter to quit.")
    while True:
        name = input("\nPlayer: ").strip()
        if not name or name.lower() in {"q", "quit", "exit"}:
            break
        predict_player(name)
