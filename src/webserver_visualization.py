import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from pyspark.sql.functions import col, count, desc, sum as spark_sum

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'visualizations')


VALID_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH", "CONNECT", "TRACE", "PROPFIND"]


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[viz] Saved → {path}")


def comma_fmt(ax_obj):
    """Apply comma-formatted integers to Y-axis (removes scientific notation)."""
    ax_obj.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))


def comma_fmt_x(ax_obj):
    """Apply comma-formatted integers to X-axis."""
    ax_obj.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))


# ══════════════════════════════════════════════════════════════
# SECTION 1 — STATUS CODE & TRAFFIC OVERVIEW
# ══════════════════════════════════════════════════════════════
def run_visualization(df_all, df_enriched, windowed, output_dir=OUTPUT_DIR):
    """
    Generates and saves all visualizations to output_dir.
    Takes df_all, df_enriched, and windowed as input for different plots.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. Status code distribution — log scale (status 200 dominates)
    status_pd = (
        df_all.groupBy("status_code").count()
        .orderBy(desc("count")).limit(15).toPandas()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(status_pd["status_code"].astype(str), status_pd["count"], color="steelblue")
    ax.set_yscale("log")
    ax.set_title("Status Code Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status Code")
    ax.set_ylabel("Request Count (log scale)")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "01_status_code_distribution.png")

    # 2. HTTP status class breakdown (pie)
    class_pd = df_all.groupBy("status_class").count().toPandas()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        class_pd["count"], labels=class_pd["status_class"],
        autopct="%1.1f%%", startangle=140,
        colors=["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    )
    ax.set_title("HTTP Status Class Breakdown", fontsize=14, fontweight="bold")
    save(fig, "02_status_class_pie.png")

    # 3. Bot vs Human traffic (pie)
    bot_pd = df_all.groupBy("is_bot").count().toPandas()
    bot_pd["label"] = bot_pd["is_bot"].map({0: "Human", 1: "Bot"})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        bot_pd["count"], labels=bot_pd["label"],
        autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90
    )
    ax.set_title("Bot vs Human Traffic", fontsize=14, fontweight="bold")
    save(fig, "03_bot_vs_human_pie.png")

    # 4. Traffic by data source — comma Y-axis
    source_pd = df_all.groupBy("source").count().toPandas()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(source_pd["source"], source_pd["count"], color=["#2196F3", "#FF9800"])
    ax.set_title("Request Count by Data Source", fontsize=14, fontweight="bold")
    ax.set_xlabel("Data Source")
    ax.set_ylabel("Request Count")
    comma_fmt(ax)
    save(fig, "04_traffic_by_source.png")

    # 5. HTTP method distribution — filter corrupted labels + comma Y-axis
    method_pd = (
        df_all.groupBy("method").count()
        .filter(col("method").isin(VALID_HTTP_METHODS))
        .orderBy(desc("count")).toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(method_pd["method"], method_pd["count"], color="slateblue")
    ax.set_yscale("log")
    ax.set_title("HTTP Method Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("HTTP Method")
    ax.set_ylabel("Request Count (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "05_http_method_distribution.png")


    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — TOP IPs & ENDPOINTS
    # ══════════════════════════════════════════════════════════════

    # 6. Top 10 IPs by request count — comma X-axis
    top_ips = (
        df_all.groupBy("ip").count()
        .orderBy(desc("count")).limit(10).toPandas()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_ips["ip"][::-1], top_ips["count"][::-1], color="darkorange")
    ax.set_title("Top 10 IPs by Request Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Request Count")
    comma_fmt_x(ax)
    save(fig, "06_top_10_ips.png")

    # 7. Top 10 requested endpoints — comma X-axis
    top_ep = (
        df_all.groupBy("endpoint").count()
        .orderBy(desc("count")).limit(10).toPandas()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_ep["endpoint"][::-1], top_ep["count"][::-1], color="mediumpurple")
    ax.set_title("Top 10 Requested Endpoints", fontsize=14, fontweight="bold")
    ax.set_xlabel("Request Count")
    comma_fmt_x(ax)
    save(fig, "07_top_10_endpoints.png")

    # 8. Top 10 IPs generating errors — comma X-axis
    top_err_ips = (
        df_all.filter(col("is_error") == 1)
        .groupBy("ip").count()
        .orderBy(desc("count")).limit(10).toPandas()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_err_ips["ip"][::-1], top_err_ips["count"][::-1], color="crimson")
    ax.set_title("Top 10 IPs Generating Errors", fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Count")
    comma_fmt_x(ax)
    save(fig, "08_top_10_error_ips.png")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — TEMPORAL PATTERNS  (df_enriched)
# ══════════════════════════════════════════════════════════════

    # 9. Request volume by hour — fill from min not 0 + comma Y-axis
    hourly_pd = (
        df_enriched.groupBy("hour_of_day").count()
        .orderBy("hour_of_day").toPandas()
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hourly_pd["hour_of_day"], hourly_pd["count"], marker="o", color="teal", linewidth=2)
    ax.fill_between(hourly_pd["hour_of_day"], hourly_pd["count"],
                    hourly_pd["count"].min(), alpha=0.2, color="teal")
    ax.set_title("Request Volume by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour (0 – 23)")
    ax.set_ylabel("Request Count")
    ax.set_xticks(range(0, 24))
    comma_fmt(ax)
    save(fig, "09_requests_by_hour.png")

    # 10. Request volume by day of week — comma Y-axis
    dow_pd = (
        df_enriched.groupBy("day_of_week").count()
        .orderBy("day_of_week").toPandas()
    )
    dow_labels = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
    dow_pd["day_name"] = dow_pd["day_of_week"].map(dow_labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow_pd["day_name"], dow_pd["count"], color="cornflowerblue")
    ax.set_title("Request Volume by Day of Week", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Request Count")
    comma_fmt(ax)
    save(fig, "10_requests_by_day_of_week.png")

    # 11. Error rate by hour — fill from min + percent Y-axis (already clean)
    hourly_err = (
        df_enriched.groupBy("hour_of_day")
        .agg(count("*").alias("total"), spark_sum("is_error").alias("errors"))
        .orderBy("hour_of_day").toPandas()
    )
    hourly_err["error_rate"] = hourly_err["errors"] / hourly_err["total"]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hourly_err["hour_of_day"], hourly_err["error_rate"],
            marker="o", color="crimson", linewidth=2)
    ax.fill_between(hourly_err["hour_of_day"], hourly_err["error_rate"],
                    hourly_err["error_rate"].min(), alpha=0.2, color="crimson")
    ax.set_title("Error Rate by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour (0 – 23)")
    ax.set_ylabel("Error Rate")
    ax.set_xticks(range(0, 24))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    save(fig, "11_error_rate_by_hour.png")

    # 12. Off-hours vs business hours (pie — already clean)
    off_pd = df_enriched.groupBy("is_off_hours").count().toPandas()
    off_pd["label"] = off_pd["is_off_hours"].map({0: "Business Hours (8–22)", 1: "Off Hours"})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        off_pd["count"], labels=off_pd["label"],
        autopct="%1.1f%%", colors=["#1565C0", "#B71C1C"], startangle=90
    )
    ax.set_title("Off-Hours vs Business Hours Traffic", fontsize=14, fontweight="bold")
    save(fig, "12_off_hours_vs_business.png")


    # ══════════════════════════════════════════════════════════════
    # SECTION 4 — ERROR ANALYSIS  (df_enriched)
    # ══════════════════════════════════════════════════════════════

    # 13. Error severity — filter "none" (not an error), reorder, comma Y-axis
    sev_order = ["low", "medium", "fatal"]
    sev_color_map = {"fatal": "#D32F2F", "medium": "#F57C00", "low": "#FBC02D"}

    sev_pd = (
        df_enriched.groupBy("error_severity").count()
        .filter(col("error_severity") != "none")
        .toPandas()
    )
    sev_pd["error_severity"] = pd.Categorical(sev_pd["error_severity"], categories=sev_order, ordered=True)
    sev_pd = sev_pd.sort_values("error_severity")
    bar_colors = [sev_color_map.get(s, "gray") for s in sev_pd["error_severity"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sev_pd["error_severity"], sev_pd["count"], color=bar_colors)
    ax.set_title("Error Severity Distribution\n(excluding normal traffic)",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Request Count")
    comma_fmt(ax)
    save(fig, "13_error_severity_distribution.png")

    # 14. Normal vs Error by source — fix axis label + comma Y-axis
    err_src = df_all.groupBy("source", "is_error").count().toPandas()
    pivot_err = err_src.pivot(index="source", columns="is_error", values="count").fillna(0)
    pivot_err.columns = ["Normal", "Error"]

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot_err.plot(kind="bar", ax=ax, color=["#43A047", "#E53935"], stacked=True)
    ax.set_title("Normal vs Error Requests by Source", fontsize=14, fontweight="bold")
    ax.set_xlabel("Data Source")
    ax.set_ylabel("Request Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Request Type")
    comma_fmt(ax)
    save(fig, "14_error_by_source_stacked.png")


    # ══════════════════════════════════════════════════════════════
    # SECTION 5 — ANOMALY ANALYSIS  (windowed)
    # ══════════════════════════════════════════════════════════════

    # 15. Anomaly type — filter "normal", comma Y-axis
    anom_pd = (
        windowed.groupBy("anomaly_type").count()
        .filter(col("anomaly_type") != "normal")
        .orderBy(desc("count")).toPandas()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(anom_pd["anomaly_type"], anom_pd["count"], color="tomato")
    ax.set_title("Anomaly Type Distribution\n(excluding normal windows)",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Anomaly Type")
    ax.set_ylabel("Window Count")
    ax.tick_params(axis="x", rotation=20)
    comma_fmt(ax)
    save(fig, "15_anomaly_type_distribution.png")

    # 16. Severity level — bar chart instead of pie (0.4% slice was unreadable)
    sev2_pd = windowed.groupBy("severity").count().toPandas()
    sev2_colors = {"high": "#C62828", "medium": "#EF6C00", "low": "#558B2F"}
    sev_order2 = ["low", "medium", "high"]
    sev2_pd["severity"] = pd.Categorical(sev2_pd["severity"], categories=sev_order2, ordered=True)
    sev2_pd = sev2_pd.sort_values("severity")
    bar_colors2 = [sev2_colors.get(s, "gray") for s in sev2_pd["severity"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sev2_pd["severity"], sev2_pd["count"], color=bar_colors2)
    ax.set_yscale("log")
    ax.set_title("Anomaly Severity Level Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Window Count (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "16_severity_level_bar.png")

    # 17. Severity score histogram — comma Y-axis
    score_pd = windowed.select("severity_score").toPandas()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(score_pd["severity_score"].dropna(), bins=40,
            color="darkcyan", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Severity Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Severity Score  (0 = normal → 1 = critical)")
    ax.set_ylabel("Window Count (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "17_severity_score_histogram.png")

    # 18. Request count per window — log scale + comma Y-axis
    req_pd = windowed.select("request_count").toPandas()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(req_pd["request_count"].dropna(), bins=50,
            color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Request Count per IP–Minute Window\n(traffic spike detection)",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Requests per Window")
    ax.set_ylabel("Window Count (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "18_request_count_per_window.png")

    # 19. Error rate distribution — log scale + comma Y-axis
    err_pd = windowed.select("error_rate").toPandas()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_pd["error_rate"].dropna(), bins=40,
            color="salmon", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Error Rate Distribution Across Windows", fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Rate (0 – 1)")
    ax.set_ylabel("Window Count (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    save(fig, "19_error_rate_distribution.png")

    # 20. Anomaly type × severity — filter "normal", fix color semantics + comma Y-axis
    anom_sev = (
        windowed.filter(col("anomaly_type") != "normal")
        .groupBy("anomaly_type", "severity").count().toPandas()
    )
    pivot_as = anom_sev.pivot(
        index="anomaly_type", columns="severity", values="count"
    ).fillna(0)

    sev_color_map2 = {"high": "#C62828", "medium": "#EF6C00", "low": "#558B2F"}
    bar_colors_as = [sev_color_map2.get(c, "gray") for c in pivot_as.columns]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot_as.plot(kind="bar", ax=ax, color=bar_colors_as)
    ax.set_title("Anomaly Type × Severity Breakdown\n(excluding normal windows)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Anomaly Type")
    ax.set_ylabel("Window Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Severity")
    comma_fmt(ax)
    save(fig, "20_anomaly_type_vs_severity.png")


    # ══════════════════════════════════════════════════════════════
    # SECTION 6 — KEY BUSINESS INSIGHTS SUMMARY
    # ══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("KEY BUSINESS INSIGHTS")
    print("═" * 60)

    total_rows    = df_all.count()
    total_bots    = df_all.filter(col("is_bot")   == 1).count()
    total_errors  = df_all.filter(col("is_error") == 1).count()
    total_windows = windowed.count()
    anomaly_wins  = windowed.filter(col("anomaly_type") != "normal").count()
    high_sev      = windowed.filter(col("severity") == "high").count()

    print(f"  Total requests analysed  : {total_rows:,}")
    print(f"  Bot traffic              : {total_bots:,}  ({100*total_bots/total_rows:.1f}%)")
    print(f"  Error requests           : {total_errors:,}  ({100*total_errors/total_rows:.1f}%)")
    print(f"  Total IP-minute windows  : {total_windows:,}")
    print(f"  Anomalous windows        : {anomaly_wins:,}  ({100*anomaly_wins/total_windows:.1f}%)")
    print(f"  High-severity windows    : {high_sev:,}  ({100*high_sev/total_windows:.1f}%)")

    print("\n  Anomaly type breakdown:")
    windowed.groupBy("anomaly_type").count() \
        .orderBy(desc("count")).show(truncate=False)

    print("\n  Peak error hours:")
    hourly_err_spark = (
        df_enriched.groupBy("hour_of_day")
        .agg(count("*").alias("total"), spark_sum("is_error").alias("errors"))
        .orderBy(desc("errors")).limit(5)
    )
    hourly_err_spark.show()

    print(f"\n[viz] All 20 plots saved to: {OUTPUT_DIR}/")
