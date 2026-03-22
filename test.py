import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 全局绘图设置
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

OUTDIR = Path("poster_figures")
OUTDIR.mkdir(exist_ok=True)

# =========================
# 图 1 数据
# 来源: [Cer18, Kru20, Bai20, Man21, Noj21, Ren24, Li25, Che25]
# =========================
df1_std = pd.DataFrame([
    {"study": "[Cer18]", "year": 2018, "outcome": "上肢运动功能", "metric": "SMD", "effect": 0.79, "note": "RCT汇总"},
    {"study": "[Kru20]", "year": 2020, "outcome": "上肢运动功能", "metric": "SMD", "effect": 0.39, "note": "EEG MI BCI"},
    {"study": "[Kru20]", "year": 2020, "outcome": "脑功能恢复", "metric": "SMD", "effect": 1.11, "note": "神经生理结局"},
    {"study": "[Bai20]", "year": 2020, "outcome": "即刻上肢功能", "metric": "SMD", "effect": 0.42, "note": "干预后"},
    {"study": "[Bai20]", "year": 2020, "outcome": "长期上肢功能", "metric": "SMD", "effect": 0.12, "note": "长期效应不显著"},
    {"study": "[Man21]", "year": 2021, "outcome": "短期上肢功能", "metric": "Hedge's g", "effect": 0.73, "note": "RCT汇总"},
    {"study": "[Man21]", "year": 2021, "outcome": "长期上肢功能", "metric": "Hedge's g", "effect": 0.33, "note": "长期效应较弱"},
    {"study": "[Noj21]", "year": 2021, "outcome": "上肢运动恢复", "metric": "SMD", "effect": 0.48, "note": "脑活动驱动BCI"},
    {"study": "[Ren24]", "year": 2024, "outcome": "BCI-FES上肢功能", "metric": "SMD", "effect": 0.50, "note": "10项RCT"},
])

df1_md = pd.DataFrame([
    {"study": "[Li25]", "year": 2025, "outcome": "FMA-UE", "metric": "MD", "effect": 3.69, "note": "21项RCT"},
    {"study": "[Li25]", "year": 2025, "outcome": "WMFT", "metric": "MD", "effect": 5.00, "note": "整体改善"},
    {"study": "[Li25]", "year": 2025, "outcome": "ARAT", "metric": "MD", "effect": 2.04, "note": "整体改善"},
    {"study": "[Che25]", "year": 2025, "outcome": "FMA-UE", "metric": "MD", "effect": 2.50, "note": "慢性期RCT"},
    {"study": "[Che25]", "year": 2025, "outcome": "MBI", "metric": "MD", "effect": 8.38, "note": "ADL改善"},
    {"study": "[Che25]", "year": 2025, "outcome": "MAL", "metric": "MD", "effect": 2.09, "note": "日常使用改善"},
    {"study": "[Che25]", "year": 2025, "outcome": "ARAT", "metric": "MD", "effect": 0.18, "note": "不显著"},
])

# =========================
# 图 2 数据
# 来源: [Man21, Ren24]
# =========================
df2_man21 = pd.DataFrame([
    {"dimension": "心理任务", "subgroup": "Movement intention", "metric": "Hedge's g", "effect": 1.21, "source": "[Man21]"},
    {"dimension": "心理任务", "subgroup": "Motor imagery", "metric": "Hedge's g", "effect": 0.55, "source": "[Man21]"},
    {"dimension": "解码特征", "subgroup": "Band power", "metric": "Hedge's g", "effect": 1.25, "source": "[Man21]"},
    {"dimension": "解码特征", "subgroup": "FBCSP", "metric": "Hedge's g", "effect": -0.23, "source": "[Man21]"},
    {"dimension": "反馈装置", "subgroup": "FES", "metric": "Hedge's g", "effect": 1.20, "source": "[Man21]"},
])

df2_ren24 = pd.DataFrame([
    {"dimension": "对照类型", "subgroup": "BCI-FES vs FES", "metric": "SMD", "effect": 0.37, "source": "[Ren24]"},
    {"dimension": "对照类型", "subgroup": "BCI-FES+常规康复 vs 常规康复", "metric": "SMD", "effect": 0.61, "source": "[Ren24]"},
    {"dimension": "病程阶段", "subgroup": "亚急性期", "metric": "SMD", "effect": 0.56, "source": "[Ren24]"},
    {"dimension": "病程阶段", "subgroup": "慢性期", "metric": "SMD", "effect": 0.42, "source": "[Ren24]"},
    {"dimension": "阈值策略", "subgroup": "调整阈值", "metric": "SMD", "effect": 0.55, "source": "[Ren24]"},
    {"dimension": "阈值策略", "subgroup": "固定阈值", "metric": "SMD", "effect": 0.43, "source": "[Ren24]"},
    {"dimension": "心理任务", "subgroup": "Motor imagery", "metric": "SMD", "effect": 0.41, "source": "[Ren24]"},
    {"dimension": "心理任务", "subgroup": "Action observation", "metric": "SMD", "effect": 0.73, "source": "[Ren24]"},
])

# =========================
# 配色函数
# =========================
def color_fig1_std(row):
    if "脑功能" in row["outcome"]:
        return "#2A9D8F"
    if "长期" in row["outcome"]:
        return "#A0AEC0"
    if "短期" in row["outcome"] or "即刻" in row["outcome"]:
        return "#4C78A8"
    return "#5B8FF9"

def color_fig1_md(row):
    if row["outcome"] == "MBI":
        return "#2A9D8F"
    if row["outcome"] == "MAL":
        return "#52B788"
    if row["outcome"] == "ARAT":
        return "#7C8DB5"
    if row["outcome"] == "WMFT":
        return "#4C78A8"
    return "#3A86FF"

def color_fig2_man21(row):
    if row["subgroup"] == "FES":
        return "#2A9D8F"
    if row["subgroup"] == "Band power":
        return "#4C78A8"
    if row["subgroup"] == "Movement intention":
        return "#52B788"
    if row["effect"] < 0:
        return "#D1495B"
    return "#7C8DB5"

def color_fig2_ren24(row):
    if row["subgroup"] == "Action observation":
        return "#2A9D8F"
    if row["subgroup"] == "调整阈值":
        return "#52B788"
    if "BCI-FES+常规康复" in row["subgroup"]:
        return "#4C78A8"
    return "#7C8DB5"

# =========================
# 图 1
# =========================
def plot_figure1():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"wspace": 0.28})

    # 面板 A
    ax = axes[0]
    dfa = df1_std.sort_values("effect", ascending=True).copy()
    dfa["label"] = dfa["study"] + "  " + dfa["outcome"]
    colors = [color_fig1_std(row) for _, row in dfa.iterrows()]
    ax.barh(dfa["label"], dfa["effect"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("A. 标准化效应量汇总", fontsize=14)
    ax.set_xlabel("效应值  SMD 或 Hedge's g", fontsize=11)
    ax.set_xlim(-0.05, 1.4)

    for i, (_, row) in enumerate(dfa.iterrows()):
        ax.text(row["effect"] + 0.03, i, f'{row["effect"]:.2f}', va="center", fontsize=10)
        ax.text(1.38, i, row["note"], va="center", ha="right", fontsize=9, color="dimgray")

    # 面板 B
    ax = axes[1]
    dfb = df1_md.sort_values("effect", ascending=True).copy()
    dfb["label"] = dfb["study"] + "  " + dfb["outcome"]
    colors = [color_fig1_md(row) for _, row in dfb.iterrows()]
    ax.barh(dfb["label"], dfb["effect"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("B. 原始量表均值差汇总", fontsize=14)
    ax.set_xlabel("效应值  MD", fontsize=11)
    ax.set_xlim(-0.2, 9.2)

    for i, (_, row) in enumerate(dfb.iterrows()):
        ax.text(row["effect"] + 0.10, i, f'{row["effect"]:.2f}', va="center", fontsize=10)
        ax.text(9.15, i, row["note"], va="center", ha="right", fontsize=9, color="dimgray")

    fig.suptitle("图 1  总体疗效证据汇总", fontsize=18, y=0.98)
    fig.text(
        0.5, 0.01,
        "注：左图为标准化效应量，右图为原始量表均值差；右图不同量表不可直接横向比较绝对数值。",
        ha="center", fontsize=10, color="dimgray"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTDIR / "figure1_overall_evidence.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "figure1_overall_evidence.svg", bbox_inches="tight")
    plt.show()

# =========================
# 图 2
# =========================
def plot_figure2():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"wspace": 0.30})

    # 面板 A: Man21
    ax = axes[0]
    dfa = df2_man21.sort_values("effect", ascending=True).copy()
    dfa["label"] = dfa["dimension"] + "｜" + dfa["subgroup"]
    colors = [color_fig2_man21(row) for _, row in dfa.iterrows()]
    ax.barh(dfa["label"], dfa["effect"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("A. 设计亚组结果  [Man21]", fontsize=14)
    ax.set_xlabel("效应值  Hedge's g", fontsize=11)
    ax.set_xlim(-0.4, 1.5)

    for i, (_, row) in enumerate(dfa.iterrows()):
        offset = 0.04 if row["effect"] >= 0 else -0.18
        ax.text(row["effect"] + offset, i, f'{row["effect"]:.2f}', va="center", fontsize=10)

    # 面板 B: Ren24
    ax = axes[1]
    dfb = df2_ren24.sort_values("effect", ascending=True).copy()
    dfb["label"] = dfb["dimension"] + "｜" + dfb["subgroup"]
    colors = [color_fig2_ren24(row) for _, row in dfb.iterrows()]
    ax.barh(dfb["label"], dfb["effect"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("B. BCI-FES 细分结果  [Ren24]", fontsize=14)
    ax.set_xlabel("效应值  SMD", fontsize=11)
    ax.set_xlim(0, 0.85)

    for i, (_, row) in enumerate(dfb.iterrows()):
        ax.text(row["effect"] + 0.02, i, f'{row["effect"]:.2f}', va="center", fontsize=10)

    fig.suptitle("图 2  哪些设计选择更可能产生更强疗效", fontsize=18, y=0.98)
    fig.text(
        0.5, 0.01,
        "结论要点：FES、movement intention 或 attempt、band power 特征，以及更强的闭环感觉后果，通常对应更大的平均效应。",
        ha="center", fontsize=10, color="dimgray"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTDIR / "figure2_design_choices.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "figure2_design_choices.svg", bbox_inches="tight")
    plt.show()

# =========================
# 运行
# =========================
if __name__ == "__main__":
    plot_figure1()
    plot_figure2()