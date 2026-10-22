"""Generate high-resolution (300 DPI) publication-quality diagrams
for the RobustLens Complete Project Guide.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_diagram_overview(output_path: Path):
    """Diagram 1: Overview Flowchart."""
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    stages = [
        {"title": "Clean Image", "sub": "Validation Sample\n(e.g., Shark, Dog)", "bg": "#e0f2fe", "border": "#0284c7"},
        {"title": "Physical Shift", "sub": "Noise, Blur, JPEG\nSeverities 1 to 5", "bg": "#fef3c7", "border": "#d97706"},
        {"title": "AI Vision Model", "sub": "CNNs & Transformers\n(ResNet, ConvNeXt, ViT, Swin)", "bg": "#f3e8ff", "border": "#7c3aed"},
        {"title": "Statistical Engine", "sub": "Accuracy, Churn, ECE\nBootstrap CIs & ANCOVA", "bg": "#dcfce7", "border": "#16a34a"},
        {"title": "Interactive Platform", "sub": "Explore, Stress Test\nLive ZeroGPU Backend", "bg": "#ffedd5", "border": "#ea580c"},
    ]

    box_w = 1.6
    box_h = 1.6
    y = 0.45

    for i, s in enumerate(stages):
        x = 0.2 + i * 1.95
        # Draw rounded rectangle
        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=s["bg"],
            edgecolor=s["border"],
            linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)

        # Text
        ax.text(x + box_w / 2, y + box_h - 0.35, s["title"],
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=3)
        ax.text(x + box_w / 2, y + 0.55, s["sub"],
                ha="center", va="center", fontsize=7.5, color="#334155", linespacing=1.2, zorder=3)

        # Arrow to next
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.32, y + box_h / 2),
                xytext=(x + box_w + 0.03, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=2, mutation_scale=15),
                zorder=1
            )

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {output_path}")


def create_diagram_architecture(output_path: Path):
    """Diagram 2: Project Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Title
    ax.text(5, 5.95, "RobustLens System Architecture & Data Flow",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    # Container 1: Client Browser (Top Left)
    c1 = patches.FancyBboxPatch((0.2, 3.4), 4.5, 2.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.8)
    ax.add_patch(c1)
    ax.text(0.4, 5.3, "Client Web Browser", fontsize=10, fontweight="bold", color="#15803d")
    ax.text(0.5, 4.7, "• User views research dashboards, charts & tables\n• User uploads images & selects stress test settings\n• Real-time reactive status indicators",
            fontsize=8, color="#166534", linespacing=1.4)

    # Container 2: Next.js Web Server on Vercel (Top Right)
    c2 = patches.FancyBboxPatch((5.3, 3.4), 4.5, 2.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#eff6ff", edgecolor="#2563eb", linewidth=1.8)
    ax.add_patch(c2)
    ax.text(5.5, 5.3, "Next.js 16 Web Server (Vercel)", fontsize=10, fontweight="bold", color="#1d4ed8")
    ax.text(5.6, 4.7, "• Static Pages: /, /order-effects, /methods, /about\n• Static Data: order_effects_data.json, benchmark_report.json\n• Secure Server API Route: /api/inference/route.ts",
            fontsize=8, color="#1e40af", linespacing=1.4)

    # Container 3: Hugging Face Spaces ZeroGPU (Bottom Right)
    c3 = patches.FancyBboxPatch((5.3, 0.4), 4.5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#faf5ff", edgecolor="#7c3aed", linewidth=1.8)
    ax.add_patch(c3)
    ax.text(5.5, 2.5, "Hugging Face Spaces (ZeroGPU Backend)", fontsize=10, fontweight="bold", color="#6d28d9")
    ax.text(5.6, 1.6, "• Gradio Client Bridge (@gradio/client) & app.py\n• Hardware: NVIDIA A10G Tensor Core GPU (24GB VRAM)\n• Models: PyTorch + timm (ResNet-50, ConvNeXt, ViT, Swin)\n• Applies pixel shift & returns Top-5 predictions in ~2.4s",
            fontsize=8, color="#5b21b6", linespacing=1.4)

    # Container 4: Offline Research Engine (Bottom Left)
    c4 = patches.FancyBboxPatch((0.2, 0.4), 4.5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#f8fafc", edgecolor="#475569", linewidth=1.8)
    ax.add_patch(c4)
    ax.text(0.4, 2.5, "Offline Research & Evaluation Engine", fontsize=10, fontweight="bold", color="#334155")
    ax.text(0.5, 1.6, "• FastAI Imagenette-v2 (N=3,925 validation images)\n• Kaggle GPU Execution: 973,400 total forward passes\n• Statistical Analysis: Clustered Bootstrap, Saturated ANCOVA\n• Precomputed Artifacts: CSVs, figures, Zenodo DOI v1.0",
            fontsize=8, color="#475569", linespacing=1.4)

    # Arrows
    # Browser <-> Vercel
    ax.annotate("", xy=(5.2, 4.8), xytext=(4.8, 4.8),
                arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.8))
    ax.text(5.0, 5.0, "HTTP / JSON", ha="center", fontsize=7, fontweight="bold", color="#0f172a")

    # Vercel <-> HF
    ax.annotate("", xy=(7.55, 2.9), xytext=(7.55, 3.3),
                arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.8))
    ax.text(7.6, 3.1, "Secure Token Proxy", ha="left", va="center", fontsize=7, fontweight="bold", color="#0f172a")

    # Offline -> Vercel (JSON upload)
    ax.annotate("", xy=(5.2, 3.5), xytext=(4.6, 2.7),
                arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.8, linestyle="dashed"))
    ax.text(4.4, 3.2, "Precomputed JSONs", ha="center", fontsize=7, fontweight="bold", color="#0f172a", rotation=35)

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {output_path}")


def create_diagram_sequence(output_path: Path):
    """Diagram 3: Live ZeroGPU Sequence Diagram."""
    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(5, 6.75, "Live ZeroGPU Inference Flow (/api/inference)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    actors = [
        {"name": "User Browser\n(Client)", "x": 1.2, "bg": "#f0fdf4", "border": "#16a34a"},
        {"name": "Next.js API\n(/api/inference)", "x": 3.7, "bg": "#eff6ff", "border": "#2563eb"},
        {"name": "Hugging Face\n(Gradio app.py)", "x": 6.3, "bg": "#faf5ff", "border": "#7c3aed"},
        {"name": "NVIDIA A10G\n(ZeroGPU CUDA)", "x": 8.8, "bg": "#fef2f2", "border": "#dc2626"},
    ]

    for a in actors:
        # Header box
        rect = patches.FancyBboxPatch((a["x"] - 0.9, 5.8), 1.8, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor=a["bg"], edgecolor=a["border"], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(a["x"], 6.15, a["name"], ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")

        # Lifeline
        ax.plot([a["x"], a["x"]], [0.4, 5.8], color="#cbd5e1", linestyle="--", lw=1.2, zorder=0)

    # Steps
    steps = [
        {"y": 5.2, "from": 1.2, "to": 3.7, "text": "1. POST /api/inference {image: base64, model: 'all', shift, severity}", "color": "#2563eb"},
        {"y": 4.6, "self": 3.7, "text": "2. Validate Base64, resolve aliases & read HF_TOKEN", "color": "#1e40af"},
        {"y": 4.0, "from": 3.7, "to": 6.3, "text": "3. Client.predict('/run_inference', [blob, model, shift, severity])", "color": "#7c3aed"},
        {"y": 3.4, "from": 6.3, "to": 8.8, "text": "4. Spawns @spaces.GPU worker & allocates CUDA memory", "color": "#dc2626"},
        {"y": 2.8, "self": 8.8, "text": "5. Applies corruption, normalizes tensor, forward pass & softmax", "color": "#991b1b"},
        {"y": 2.2, "from": 8.8, "to": 6.3, "text": "6. Returns Top-5 class labels, probabilities & latency", "color": "#dc2626"},
        {"y": 1.6, "from": 6.3, "to": 3.7, "text": "7. Clean prediction JSON payload returned to API route", "color": "#7c3aed"},
        {"y": 1.0, "from": 3.7, "to": 1.2, "text": "8. HTTP 200 JSON {success: true, results: [...], latency: 2.41s}", "color": "#16a34a"},
    ]

    for s in steps:
        y = s["y"]
        if "from" in s:
            x1, x2 = s["from"], s["to"]
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color=s["color"], lw=1.5, mutation_scale=12))
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y + 0.16, s["text"], ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0f172a",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#e2e8f0", alpha=0.9))
        else:
            x = s["self"]
            rect = patches.FancyBboxPatch((x + 0.1, y - 0.18), 2.2, 0.36, boxstyle="round,pad=0.04",
                                         facecolor="#f8fafc", edgecolor=s["color"], linewidth=1)
            ax.add_patch(rect)
            ax.text(x + 1.2, y, s["text"], ha="center", va="center", fontsize=7, color="#1e293b")

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {output_path}")


def create_diagram_pipeline(output_path: Path):
    """Diagram 4: Offline Research Pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    ax.text(5, 3.8, "RobustLens 973,400-Inference Offline Research Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    columns = [
        {"title": "1. Dataset Ingestion", "w": 2.1, "items": ["FastAI Imagenette-v2", "13,394 Total Images", "Calibration (N=9,469)", "Validation (N=3,925)"], "bg": "#f0fdf4", "border": "#16a34a"},
        {"title": "2. Permutation Grid", "w": 2.1, "items": ["6 Sequences of N, B, J", "Nominal (Fixed Shift)", "Calibrated Fidelity", "(Matched PSNR/SSIM)"], "bg": "#eff6ff", "border": "#2563eb"},
        {"title": "3. Vision Models", "w": 2.1, "items": ["ResNet-50 (CNN)", "ConvNeXt-Tiny (Modern)", "ViT-Base/16 (Patch Attn)", "Swin-Tiny (Shifted Win)"], "bg": "#faf5ff", "border": "#7c3aed"},
        {"title": "4. Statistical Engine", "w": 2.1, "items": ["Paired Bootstrap (B=5k)", "Saturated ANCOVA (det=-6)", "5-State Churn Matrix", "Calibration & Conformal"], "bg": "#fff7ed", "border": "#ea580c"},
    ]

    for i, col in enumerate(columns):
        x = 0.4 + i * 2.35
        # Draw box
        rect = patches.FancyBboxPatch((x, 0.4), 2.1, 3.0, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor=col["bg"], edgecolor=col["border"], linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + 1.05, 3.15, col["title"], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")

        # Divider
        ax.plot([x + 0.2, x + 1.9], [2.85, 2.85], color=col["border"], lw=1)

        # Items
        for j, item in enumerate(col["items"]):
            ax.text(x + 0.2, 2.5 - j * 0.52, f"• {item}", fontsize=8, color="#1e293b")

        # Arrow
        if i < len(columns) - 1:
            ax.annotate("", xy=(x + 2.3, 1.9), xytext=(x + 2.12, 1.9),
                        arrowprops=dict(arrowstyle="->", color="#64748b", lw=2, mutation_scale=14))

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {output_path}")


def create_diagram_timeline(output_path: Path):
    """Diagram 5: Project Milestones Timeline."""
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    ax.text(5.5, 3.15, "RobustLens Research & Engineering Milestones",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    phases = [
        {"phase": "Phase 1", "title": "Scaffolding", "detail": "uv Python ML &\nNext.js Monorepo", "x": 1.0},
        {"phase": "Phase 2", "title": "Baselines", "detail": "Speed Benchmarks &\nTheme Engine", "x": 2.8},
        {"phase": "Phase 3", "title": "Kaggle Runs", "detail": "973k Inferences on\nImagenette-v2", "x": 4.6},
        {"phase": "Phase 4", "title": "Paper & Stats", "detail": "Zenodo DOI v1.0 &\nBootstrap Suite", "x": 6.4},
        {"phase": "Phase 5", "title": "Web Platform", "detail": "HeroUI v3, Charts &\n9 Master Routes", "x": 8.2},
        {"phase": "Phase 6", "title": "Production", "detail": "Vercel Deploy &\nHF ZeroGPU Live", "x": 10.0},
    ]

    # Timeline bar
    ax.plot([1.0, 10.0], [1.7, 1.7], color="#2563eb", lw=3.5, zorder=1)

    for p in phases:
        x = p["x"]
        # Node circle
        circle = plt.Circle((x, 1.7), 0.16, color="#1e40af", ec="white", lw=2, zorder=3)
        ax.add_patch(circle)

        # Phase label above
        ax.text(x, 2.05, p["phase"], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2563eb")
        ax.text(x, 2.38, p["title"], ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

        # Detail below
        box_w = 1.55
        rect = patches.FancyBboxPatch((x - box_w/2, 0.4), box_w, 0.95, boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 0.88, p["detail"], ha="center", va="center", fontsize=7.2, color="#1e293b",
                linespacing=1.25)

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = base_dir / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    create_diagram_overview(figures_dir / "diagram1_overview.png")
    create_diagram_architecture(figures_dir / "diagram2_architecture.png")
    create_diagram_sequence(figures_dir / "diagram3_sequence.png")
    create_diagram_pipeline(figures_dir / "diagram4_pipeline.png")
    create_diagram_timeline(figures_dir / "diagram5_timeline.png")
    print("All 5 high-resolution diagrams generated successfully!")


if __name__ == "__main__":
    main()
