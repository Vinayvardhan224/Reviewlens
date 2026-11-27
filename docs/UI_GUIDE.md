# ReviewLens Glassmorphism & Bento Grid UI Guide

## 1. Executive Summary & Aesthetic Identity
ReviewLens employs a **Glassmorphism Design System** arranged in an asymmetrical **Bento Grid Layout**, specifically engineered for high-density NLP analytics while preserving a cohesive, modern dark-purple aesthetic.

### Core Visual Principles
- **Atmospheric Aurora Base**: Deep cosmic violet (`#07051A`) anchored with subtle, fixed radial blur blobs (pointer-events disabled to eliminate layout thrashing).
- **Frosted Translucent Tiles**: Semi-transparent surface containers (`rgba(22, 17, 46, 0.62)`) with backdrop blur (`14px`) and delicate luminescent borders (`rgba(255, 255, 255, 0.10)`).
- **Semantic-Only Color Coding**: Hue conveys diagnostic meaning exclusively:
  - **Positive / Genuine**: Emerald Green (`#34D399`)
  - **Negative / Spam**: Coral Red (`#F87171`)
  - **Mixed / Warning**: Amber Gold (`#FBBF24`)
  - **Accent / Interactive**: Cyan Blue (`#22D3EE`) & Electric Violet (`#9B5CFF`)
- **Strict WCAG AA Contrast**: High-contrast typography (`#F1F5F9`) ensuring >= 4.5:1 contrast against all frosted glass tile backgrounds.

---

## 2. Design Tokens & CSS Variables

Defined centrally in `components/theme.py`:

```css
:root {
  --bg: #07051A;
  --bg2: #0D0A24;
  --glass: rgba(22, 17, 46, 0.62);
  --glass-strong: rgba(30, 23, 62, 0.78);
  --border: rgba(255, 255, 255, 0.10);
  --border-hover: rgba(155, 92, 255, 0.55);
  --primary: #7138F5;
  --primary-2: #9B5CFF;
  --accent: #22D3EE;
  --text: #F1F5F9;
  --muted: #A5A3C2;
  --pos: #34D399;
  --neg: #F87171;
  --mixed: #FBBF24;
  --radius: 18px;
  --blur: 14px;
}
```

---

## 3. Reusable Component Primitives

### 3.1 Frosted Glass Cards (`.rl-glass` & `glass_card()`)
Used for all analytic modules, data cards, and interactive controls:
- **Hover Micro-interaction**: `transform: translateY(-2px)` with subtle violet glow (`box-shadow: 0 14px 35px rgba(113, 56, 245, 0.35)`).
- **Reduced Motion Support**: Disables transitions and transforms under `prefers-reduced-motion: reduce`.

### 3.2 High-Contrast Bento KPI Tiles (`kpi_tile()`)
Constructed with tabular numbers (`font-variant-numeric: tabular-nums`) to prevent jitter during real-time filtering:
- Big prominent numerical display (1.95rem / 800 weight)
- Uppercase metric label with tracked spacing
- Semantic delta chip (`.rl-chip-pos`, `.rl-chip-neg`, `.rl-chip-mixed`, `.rl-chip-neutral`)

### 3.3 Harmonized Plotly Charts (`apply_glass_layout()`)
Ensures all data visualizations integrate seamlessly into frosted glass:
- `paper_bgcolor: "rgba(0,0,0,0)"`, `plot_bgcolor: "rgba(0,0,0,0)"`
- Font: `'Plus Jakarta Sans'` with `#F1F5F9` text
- Subtle gridlines: `rgba(255, 255, 255, 0.08)`
- Standard Palette: `["#9B5CFF", "#22D3EE", "#34D399", "#FBBF24", "#F87171", "#F472B6"]`

---

## 4. Key Page Bento Grid Architectures

### 4.1 Review Intelligence
- **Hero Tile**: Headline, product thumbnail, active filter tags, and quick-search entry.
- **KPI Quad Grid**: Reviews Analysed, Net Sentiment Score (NSS), Fake Review Rate %, Top Aspect Driver.
- **Wide Chart Tile**: Aspect-based sentiment breakdown bar chart.
- **Tall Side Tile**: Unsupervised LDA topic clusters with interactive keyword pills.
- **Bottom Alert Tile**: AI-generated managerial summary and critical customer pain points.

### 4.2 Live Diagnostic Lab (Review Analyzer)
- **Left Hero Tile**: Live multiline review input + real-time syntactic word highlighting (`components/highlight.py`).
- **Right KPI Column**: Compound sentiment gauge, confidence rating, spam probability score, extracted aspect count.
- **Lower Bento Tier**: Extracted aspect sentiment meters + generative AI reply recommendation.

### 4.3 Model Evaluation & Benchmarks
- **Top Metric Cards**: Accuracy and Macro F1 for VADER, Classical ML (TF-IDF + LogReg), and SemEval aspect parsing.
- **Confusion Matrix Grid**: Symmetrical confusion matrix heatmaps comparing predicted vs true classes.
- **Limitations Bento**: Prominently framed callout detailing domain shift, small demo sizes, and syntactic constraints.

### 4.4 Product Duel & Comparison
- **Flanking Competitor Tiles**: Left and right glass cards showcasing selected product specifications and NSS scores.
- **Center Radar Tile**: Multi-axial aspect radar chart comparing battery, camera, display, performance, and build quality.

---

## 5. Mobile & Accessibility Engineering
- **Breakpoints**: At `<= 768px`, all multi-column layouts gracefully collapse into a single-column stack targeting both `div[data-testid="column"]` and `div[data-testid="stColumn"]`.
- **Zero Horizontal Overflow**: Guaranteed at viewport widths as narrow as `390px`.
- **Touch Target Standard**: All interactive inputs, buttons, and radio pills enforce a minimum height of `44px`.
- **Low-Power / Performance Toggle**: Sidebar contains an "Appearance & Performance" expander enabling the "Reduce visual effects" mode, stripping backdrop filters for instantaneous rendering on low-end hardware.
