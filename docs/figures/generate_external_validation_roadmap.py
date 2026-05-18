"""
Generate External Validation Roadmap figure for Slide 14.
All text in English.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = "docs/figures"

def draw_external_validation_roadmap():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(6, 8.6, "External Validation Roadmap for Stroke MI-EEG Decoding",
            fontsize=16, fontweight='bold', ha='center', va='top', color='#1a1a1a')
    ax.text(6, 8.15, "Beyond Liu24 (XWStroke): Public datasets are scarce — validate strategically",
            fontsize=11, ha='center', va='top', color='#555555', style='italic')

    # ============================================
    # TIER 1 HEADER
    # ============================================
    tier1_header = FancyBboxPatch((0.5, 7.1), 11, 0.7, boxstyle="round,pad=0.05,rounding_size=0.15",
                                   facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2.5)
    ax.add_patch(tier1_header)
    ax.text(6, 7.45, "TIER 1  —  PUBLICLY AVAILABLE  (Immediate Use)",
            fontsize=12, fontweight='bold', ha='center', va='center', color='#1565c0')

    # ============================================
    # TIER 1 BOXES
    # ============================================

    # Box 1: Thi25
    thi_color = '#c62828'
    thi_bg = '#ffebee'
    thi_box = FancyBboxPatch((0.5, 5.0), 3.5, 1.9, boxstyle="round,pad=0.03,rounding_size=0.12",
                              facecolor=thi_bg, edgecolor=thi_color, linewidth=2.2)
    ax.add_patch(thi_box)
    ax.text(2.25, 6.55, "Thi25", fontsize=13, fontweight='bold', ha='center', va='top', color=thi_color)
    ax.text(2.25, 6.25, "N = 30  |  Post-acute", fontsize=9, ha='center', va='top', color='#444444')
    ax.text(2.25, 5.85, "4-class MI (all limbs)", fontsize=9, ha='center', va='top', color='#666666')
    ax.text(2.25, 5.55, "GitHub  |  128 Hz  |  Emotiv",
            fontsize=8.5, ha='center', va='top', color='#777777')
    # Badge
    badge1 = FancyBboxPatch((0.7, 5.05), 3.1, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=thi_color, edgecolor='none')
    ax.add_patch(badge1)
    ax.text(2.25, 5.26, "RICHEST CLINICAL INFO", fontsize=8, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(2.25, 5.0, "NIHSS · Lesion · mRS · Oxford", fontsize=7.5,
            ha='center', va='top', color=thi_color, style='italic')

    # Box 2: Liu25
    liu_color = '#1565c0'
    liu_bg = '#e3f2fd'
    liu_box = FancyBboxPatch((4.25, 5.0), 3.5, 1.9, boxstyle="round,pad=0.03,rounding_size=0.12",
                              facecolor=liu_bg, edgecolor=liu_color, linewidth=2.2)
    ax.add_patch(liu_box)
    ax.text(6.0, 6.55, "Liu25", fontsize=13, fontweight='bold', ha='center', va='top', color=liu_color)
    ax.text(6.0, 6.25, "N = 27  |  Recovery (1–12 mo)", fontsize=9, ha='center', va='top', color='#444444')
    ax.text(6.0, 5.85, "Lower-limb gait MI vs idle", fontsize=9, ha='center', va='top', color='#666666')
    ax.text(6.0, 5.55, "Figshare  |  BIDS  |  Multi-paradigm",
            fontsize=8.5, ha='center', va='top', color='#777777')
    badge2 = FancyBboxPatch((4.45, 5.05), 3.1, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=liu_color, edgecolor='none')
    ax.add_patch(badge2)
    ax.text(6.0, 5.26, "CLOSEST TO LIU24 STRUCTURE", fontsize=8, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(6.0, 5.0, "Affected side · Lesion area · Duration", fontsize=7.5,
            ha='center', va='top', color=liu_color, style='italic')

    # Box 3: Cho21
    cho_color = '#2e7d32'
    cho_bg = '#e8f5e9'
    cho_box = FancyBboxPatch((8.0, 5.0), 3.5, 1.9, boxstyle="round,pad=0.03,rounding_size=0.12",
                              facecolor=cho_bg, edgecolor=cho_color, linewidth=2.2)
    ax.add_patch(cho_box)
    ax.text(9.75, 6.55, "Cho21", fontsize=13, fontweight='bold', ha='center', va='top', color=cho_color)
    ax.text(9.75, 6.25, "N = 10  |  ~11 mo post-stroke", fontsize=9, ha='center', va='top', color='#444444')
    ax.text(9.75, 5.85, "L/R hand grasp attempt", fontsize=9, ha='center', va='top', color='#666666')
    ax.text(9.75, 5.55, "Challenge benchmark  |  80 train / 40 test",
            fontsize=8.5, ha='center', va='top', color='#777777')
    badge3 = FancyBboxPatch((8.2, 5.05), 3.1, 0.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=cho_color, edgecolor='none')
    ax.add_patch(badge3)
    ax.text(9.75, 5.26, "CROSS-SUBJECT BENCHMARK", fontsize=8, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(9.75, 5.0, "Affected side · Age · Gender", fontsize=7.5,
            ha='center', va='top', color=cho_color, style='italic')

    # ============================================
    # TIER 2 HEADER
    # ============================================
    tier2_header = FancyBboxPatch((0.5, 3.9), 11, 0.55, boxstyle="round,pad=0.05,rounding_size=0.15",
                                   facecolor='#f5f5f5', edgecolor='#757575', linewidth=2)
    ax.add_patch(tier2_header)
    ax.text(6, 4.17, "TIER 2  —  COLLABORATION REQUIRED  (Data Access Application)",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#424242')

    # ============================================
    # TIER 2 BOXES
    # ============================================
    tier2_items = [
        ("Man22", "N = 136", "FMA-rich", "Contralesional BCI"),
        ("Seb20b", "N = 36", "Lesion stratification", "FMA-UE/LE · BI · MAS"),
        ("Fro17b", "N = 74", "Multicenter RCT", "FMMA · ARAT · Lesion"),
        ("Par16", "N = 12", "SM1+ / SM1- / INF", "FMA-UE · Lesion anatomy"),
        ("Shu18b", "N = 24", "BCI inefficiency", "FMA-UE · Cortical/Subcortical"),
    ]

    box_width = 2.0
    start_x = 0.7
    gap = 0.15
    y_pos = 3.2
    for i, (name, n, strength, scales) in enumerate(tier2_items):
        x = start_x + i * (box_width + gap)
        t2_box = FancyBboxPatch((x, y_pos - 0.65), box_width, 0.85,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor='#fafafa', edgecolor='#9e9e9e', linewidth=1.2)
        ax.add_patch(t2_box)
        ax.text(x + box_width/2, y_pos - 0.08, name, fontsize=9.5, fontweight='bold',
                ha='center', va='top', color='#424242')
        ax.text(x + box_width/2, y_pos - 0.32, n, fontsize=8, ha='center', va='top', color='#616161')
        ax.text(x + box_width/2, y_pos - 0.52, strength, fontsize=7, ha='center', va='top', color='#888888')

    # ============================================
    # RECOMMENDED SEQUENCE (Bottom flow)
    # ============================================
    # Background strip
    seq_bg = FancyBboxPatch((0.5, 0.8), 11, 1.6, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor='#fff8e1', edgecolor='#f9a825', linewidth=2)
    ax.add_patch(seq_bg)
    ax.text(6, 2.15, "RECOMMENDED VALIDATION SEQUENCE", fontsize=12, fontweight='bold',
            ha='center', va='center', color='#e65100')

    # Flow nodes
    flow_y = 1.35
    node_h = 0.5
    nodes = [
        (1.8, "Thi25", "Validate clinical\nstratification", '#c62828'),
        (4.5, "Liu25", "Validate cross-task\nrobustness", '#1565c0'),
        (7.2, "Cho21", "Validate cross-subject\ntransfer", '#2e7d32'),
        (9.8, "Tier-2", "Scale-up\ncollaboration", '#757575'),
    ]

    for i, (cx, label, desc, color) in enumerate(nodes):
        node = FancyBboxPatch((cx - 0.9, flow_y - node_h/2), 1.8, node_h,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(node)
        ax.text(cx, flow_y + 0.08, label, fontsize=10, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(cx, flow_y - 0.15, desc, fontsize=7.5, ha='center', va='center', color='#555555')

        if i < len(nodes) - 1:
            next_cx = nodes[i+1][0]
            ax.annotate("", xy=(next_cx - 0.95, flow_y), xytext=(cx + 0.95, flow_y),
                        arrowprops=dict(arrowstyle="->", color='#f9a825', lw=2))

    # ============================================
    # BOTTOM BANNER
    # ============================================
    ax.text(6, 0.35, "Even one Tier-1 validation (especially Thi25) would make our work unique\n"
                      "— no prior Liu24-based paper has reported cross-dataset validation.",
            fontsize=9.5, fontweight='bold', ha='center', va='center', color='#bf360c',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffecb3', edgecolor='#f9a825', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_external_validation_roadmap.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_external_validation_roadmap.png")


if __name__ == "__main__":
    draw_external_validation_roadmap()
    print("Done!")
