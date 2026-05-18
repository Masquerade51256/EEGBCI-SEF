"""
Generate remaining PPT figures for v3 outline:
1. fig_stroke_heterogeneity.png (Slide 2)
2. fig_clinical_adaptive_framework.png (Slide 13)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Wedge, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = "docs/figures"


def draw_stroke_heterogeneity():
    """Slide 2: Stroke Heterogeneity & Clinical Measurement infographic"""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, "Stroke Heterogeneity: Why One-Size-Fits-All Fails",
            fontsize=16, fontweight='bold', ha='center', va='top', color='#1a1a1a')
    ax.text(6, 6.35, "Acute stroke patients vary across multiple clinical dimensions, creating fundamentally different EEG patterns",
            fontsize=10.5, ha='center', va='top', color='#555555', style='italic')

    # Central brain icon (simplified)
    brain = Ellipse((6, 4.0), 2.2, 1.6, facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2.5)
    ax.add_patch(brain)
    ax.text(6, 4.0, "Stroke\nLesion", fontsize=11, fontweight='bold', ha='center', va='center', color='#1565c0')
    # Lesion marks
    ax.add_patch(Ellipse((5.6, 4.2), 0.4, 0.3, facecolor='#ef9a9a', edgecolor='#c62828', linewidth=1.5, alpha=0.7))
    ax.add_patch(Ellipse((6.4, 3.7), 0.35, 0.25, facecolor='#ef9a9a', edgecolor='#c62828', linewidth=1.5, alpha=0.5))

    # Four factor boxes around the brain
    factors = [
        (1.5, 5.2, "Lesion Location", "Cortical\nSubcortical\nBrainstem\nMixed", '#c62828', '#ffebee'),
        (10.5, 5.2, "NIHSS Severity", "Light (1–3)\nMedium (4–7)\nSevere (≥8)", '#e65100', '#fff3e0'),
        (1.5, 2.3, "Paralysis Side", "Left Hemiplegia\nRight Hemiplegia\nBilateral", '#2e7d32', '#e8f5e9'),
        (10.5, 2.3, "Disease Duration", "Acute (≤3 mo)\nChronic (>3 mo)\nLongitudinal", '#1565c0', '#e3f2fd'),
    ]

    for cx, cy, title, body, edge_c, bg_c in factors:
        box = FancyBboxPatch((cx - 1.3, cy - 1.0), 2.6, 2.0,
                              boxstyle="round,pad=0.03,rounding_size=0.15",
                              facecolor=bg_c, edgecolor=edge_c, linewidth=2)
        ax.add_patch(box)
        ax.text(cx, cy + 0.55, title, fontsize=10, fontweight='bold',
                ha='center', va='center', color=edge_c)
        ax.text(cx, cy - 0.15, body, fontsize=8.5, ha='center', va='center', color='#444444')

    # Arrows from factors to brain
    arrow_style = dict(arrowstyle='->', color='#9e9e9e', lw=1.5, connectionstyle='arc3,rad=0.1')
    ax.annotate("", xy=(5.0, 4.6), xytext=(2.5, 5.0), arrowprops=arrow_style)
    ax.annotate("", xy=(7.0, 4.6), xytext=(9.5, 5.0), arrowprops=arrow_style)
    ax.annotate("", xy=(5.0, 3.4), xytext=(2.5, 2.8), arrowprops=arrow_style)
    ax.annotate("", xy=(7.0, 3.4), xytext=(9.5, 2.8), arrowprops=arrow_style)

    # Bottom: Clinical Scales bar
    scale_bg = FancyBboxPatch((0.8, 0.3), 10.4, 1.2,
                               boxstyle="round,pad=0.03,rounding_size=0.1",
                               facecolor='#fafafa', edgecolor='#757575', linewidth=1.5)
    ax.add_patch(scale_bg)
    ax.text(6, 1.25, "Clinical Scales Quantify Heterogeneity", fontsize=10.5, fontweight='bold',
            ha='center', va='center', color='#424242')

    scales = [
        ("NIHSS", "0–42", "Neurological deficit", '#c62828'),
        ("FMA", "0–66/100", "Motor function", '#e65100'),
        ("mRS", "0–5", "Disability level", '#1565c0'),
        ("MBI", "0–100", "Daily activity", '#2e7d32'),
    ]
    x_positions = [2.0, 4.5, 7.0, 9.5]
    for (name, range_, desc, color), x in zip(scales, x_positions):
        ax.text(x, 0.8, name, fontsize=9.5, fontweight='bold', ha='center', va='center', color=color)
        ax.text(x, 0.55, range_, fontsize=8, ha='center', va='center', color='#666666')
        ax.text(x, 0.35, desc, fontsize=7, ha='center', va='center', color='#888888')

    # Bottom takeaway
    ax.text(6, 0.05, "Heterogeneity is NOT noise — it is structured clinical information that current decoders ignore",
            fontsize=9, fontweight='bold', ha='center', va='center', color='#bf360c',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffecb3', edgecolor='#f9a825', linewidth=1.2))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_stroke_heterogeneity.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_stroke_heterogeneity.png")


def draw_clinical_adaptive_framework():
    """Slide 13: Clinical Adaptive Framework architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Title
    ax.text(6, 8.2, "Route B: Clinical Adaptive Framework",
            fontsize=16, fontweight='bold', ha='center', va='top', color='#1a1a1a')
    ax.text(6, 7.85, "Exploit clinical stratification instead of erasing it",
            fontsize=11, ha='center', va='top', color='#555555', style='italic')

    # ============ INPUTS (Top) ============
    input_bg = FancyBboxPatch((0.5, 6.5), 11, 1.0,
                               boxstyle="round,pad=0.03,rounding_size=0.1",
                               facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(input_bg)
    ax.text(6, 7.25, "INPUTS", fontsize=11, fontweight='bold', ha='center', va='center', color='#2e7d32')

    # Input boxes
    input_boxes = [
        (1.8, 6.75, "EEG Signal\n[C × T]", '#4caf50'),
        (4.5, 6.75, "NIHSS\nScore", '#66bb6a'),
        (6.0, 6.75, "Lesion\nLocation", '#66bb6a'),
        (7.5, 6.75, "Paralysis\nSide", '#66bb6a'),
        (9.2, 6.75, "Disease\nDuration", '#66bb6a'),
    ]
    for cx, cy, text, color in input_boxes:
        box = FancyBboxPatch((cx - 0.7, cy - 0.35), 1.4, 0.7,
                              boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor='white', edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(cx, cy, text, fontsize=8, ha='center', va='center', color='#333333')

    # ============ TWO PATHWAYS ============

    # Path 1 header
    path1_bg = FancyBboxPatch((0.5, 4.3), 5.2, 1.8,
                               boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2.2)
    ax.add_patch(path1_bg)
    ax.text(3.1, 5.85, "APPROACH 1", fontsize=10, fontweight='bold',
            ha='center', va='center', color='#1565c0')
    ax.text(3.1, 5.55, "Dynamic Source-Domain Selection", fontsize=9.5, fontweight='bold',
            ha='center', va='center', color='#0d47a1')

    # Path 1 content
    path1_steps = [
        (3.1, 5.15, "1. Clinical Similarity Matching", '#1976d2'),
        (3.1, 4.8, "Match test patient to source subset\nby lesion / NIHSS / duration", '#555555'),
        (3.1, 4.35, "2. Personalized Model Training", '#1976d2'),
        (3.1, 4.0, "Train on matched subset only", '#555555'),
    ]
    for cx, cy, text, color in path1_steps:
        ax.text(cx, cy, text, fontsize=8, ha='center', va='center', color=color)

    # Path 2 header
    path2_bg = FancyBboxPatch((6.3, 4.3), 5.2, 1.8,
                               boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor='#fff3e0', edgecolor='#e65100', linewidth=2.2)
    ax.add_patch(path2_bg)
    ax.text(8.9, 5.85, "APPROACH 2", fontsize=10, fontweight='bold',
            ha='center', va='center', color='#e65100')
    ax.text(8.9, 5.55, "Conditional Feature Extractor", fontsize=9.5, fontweight='bold',
            ha='center', va='center', color='#bf360c')

    # Path 2 content
    path2_steps = [
        (8.9, 5.15, "1. Clinical Metadata Encoding", '#ef6c00'),
        (8.9, 4.8, "Embed NIHSS / lesion / duration\nas auxiliary vector", '#555555'),
        (8.9, 4.35, "2. Gated Feature Extraction", '#ef6c00'),
        (8.9, 4.0, "Clinical variables modulate\nbackbone feature pathways", '#555555'),
    ]
    for cx, cy, text, color in path2_steps:
        ax.text(cx, cy, text, fontsize=8, ha='center', va='center', color=color)

    # Arrows from inputs to pathways
    ax.annotate("", xy=(3.1, 6.15), xytext=(3.1, 6.5),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=2))
    ax.annotate("", xy=(8.9, 6.15), xytext=(8.9, 6.5),
                arrowprops=dict(arrowstyle='->', color='#e65100', lw=2))

    # ============ SHARED BACKBONE ============
    backbone = FancyBboxPatch((3.5, 2.9), 5.0, 1.0,
                               boxstyle="round,pad=0.03,rounding_size=0.12",
                               facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=2.2)
    ax.add_patch(backbone)
    ax.text(6, 3.65, "ADFCNN Backbone", fontsize=11, fontweight='bold',
            ha='center', va='center', color='#7b1fa2')
    ax.text(6, 3.25, "Shared feature extractor processing [B, 1, C×bands, T]",
            fontsize=8.5, ha='center', va='center', color='#555555')

    # Arrows from pathways to backbone
    ax.annotate("", xy=(4.5, 3.9), xytext=(3.5, 4.3),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.8))
    ax.annotate("", xy=(7.5, 3.9), xytext=(8.5, 4.3),
                arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.8))

    # ============ CLASSIFIER HEAD ============
    classifier = FancyBboxPatch((4.5, 1.7), 3.0, 0.9,
                                 boxstyle="round,pad=0.03,rounding_size=0.1",
                                 facecolor='#fce4ec', edgecolor='#c62828', linewidth=2)
    ax.add_patch(classifier)
    ax.text(6, 2.35, "Classifier Head", fontsize=10, fontweight='bold',
            ha='center', va='center', color='#c62828')
    ax.text(6, 2.0, "Task: Affected / Unaffected", fontsize=8.5,
            ha='center', va='center', color='#555555')

    ax.annotate("", xy=(6, 2.6), xytext=(6, 2.9),
                arrowprops=dict(arrowstyle='->', color='#7b1fa2', lw=2))

    # ============ OUTPUT ============
    output = FancyBboxPatch((4.8, 0.6), 2.4, 0.7,
                             boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(output)
    ax.text(6, 1.05, "Decoded Output", fontsize=10, fontweight='bold',
            ha='center', va='center', color='#2e7d32')
    ax.text(6, 0.75, "+ Clinical Explanation", fontsize=8,
            ha='center', va='center', color='#555555', style='italic')

    ax.annotate("", xy=(6, 1.3), xytext=(6, 1.7),
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=2))

    # ============ ADVANTAGE BANNER ============
    ax.text(6, 0.15, "Advantage: Interpretable + Clinically Actionable — model behavior traces back to patient characteristics",
            fontsize=9, fontweight='bold', ha='center', va='center', color='#1b5e20',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#c8e6c9', edgecolor='#4caf50', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_clinical_adaptive_framework.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: fig_clinical_adaptive_framework.png")


if __name__ == "__main__":
    draw_stroke_heterogeneity()
    draw_clinical_adaptive_framework()
    print("\nAll v3 figures generated successfully!")
