"""
ReviewLens Action Brief & Feedback Engineering Intelligence Module.
Inspired by Vector (vectorco.in) requirement engineering and deterministic prompt analysis:
Translates unstructured consumer feedback into structured, testable, and engineering-ready
6-Section Technical Action Briefs with Given/When/Then (BDD) Acceptance Criteria.

Core Capabilities:
1. Review Actionability & Gap Index (0-100 Score, 6 Structural Layers)
2. 6-Section Engineering Action Brief Synthesizer (PRD / Jira / Linear format)
3. Deterministic BDD (Given / When / Then) Acceptance Criteria Generator
4. Jira & Linear Markdown Formatter with 1-Click Export
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd


# 6 Structural Layers of Actionable Customer Feedback
ACTIONABILITY_LAYERS = [
    {
        "id": "subsystem",
        "name": "Subsystem / Component",
        "weight": 25,
        "description": "Specific hardware part, sensor, chip, display, driver, or UI component identified."
    },
    {
        "id": "defect_severity",
        "name": "Defect / Failure Symptom",
        "weight": 25,
        "description": "Concrete observable failure, crash, stutter, overheating, or degradation."
    },
    {
        "id": "trigger_context",
        "name": "Trigger & Usage Context",
        "weight": 20,
        "description": "Environmental conditions, operating state, or trigger action leading to defect."
    },
    {
        "id": "polarity_confidence",
        "name": "Polarity & Sentiment Valence",
        "weight": 10,
        "description": "Clear and unambiguous polarity without self-contradictory stance."
    },
    {
        "id": "reproducibility",
        "name": "Reproducibility / Frequency",
        "weight": 10,
        "description": "Frequency indicators (consistently, always, once, intermittent, after update)."
    },
    {
        "id": "expected_outcome",
        "name": "Expected Outcome / Benchmark",
        "weight": 10,
        "description": "Target expectation or comparison benchmark (e.g., expected 8 hrs, previous model)."
    }
]

# Gazetteers for Rule-Based Layer Extraction
CONTEXT_TRIGGERS = [
    "when", "while", "during", "after", "outdoors", "indoors", "under direct sun",
    "on 5g", "on wifi", "recording 4k", "charging", "gaming", "multitasking",
    "at night", "in low light", "streaming", "standby", "idle", "at 120hz",
    "heavy load", "full brightness", "compilation", "on battery"
]

REPRODUCIBILITY_MARKERS = [
    "always", "every time", "constantly", "consistently", "repeatedly",
    "often", "frequently", "intermittent", "randomly", "sometimes",
    "after the update", "since day one", "within an hour"
]

EXPECTATION_MARKERS = [
    "expected", "should be", "supposed to", "compared to", "promised",
    "claim", "better than", "worse than", "industry standard", "at this price"
]

SUBSYSTEM_MAP = {
    "Battery": "Power Management IC (PMIC) & Battery Fuel Gauge Controller",
    "Charging": "Fast Charging Controller & USB-PD Thermal Throttling Subsystem",
    "Camera": "Image Signal Processor (ISP) & Optical Image Stabilization (OIS)",
    "Display": "OLED Display Panel & Refresh Rate Controller Driver (DDIC)",
    "Performance": "SoC DVFS Governor, Thermal Throttler & Memory Controller",
    "Processor": "CPU Multi-Core Governor & DVFS Voltage Scaling Driver",
    "RAM": "LPDDR Memory Management & Background Process Killer (LowMemoryKiller)",
    "Cooling": "Vapor Chamber Heat Sink & Thermal Sensor Bus",
    "Sound quality": "Audio Codec DAC & Low-Frequency Transducer Enclosure",
    "Build Quality": "Chassis Structural Rigidity & Anodized Finish Treatment",
    "Value for Money": "BOM Cost-to-Feature Margin Alignment"
}


def evaluate_review_actionability(
    text: str,
    aspects: Optional[List[str]] = None,
    sentiment_compound: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates customer feedback on its engineering actionability (0-100 Score).
    Measures coverage across 6 structural layers and generates explicit gap findings.
    """
    if not text or not text.strip():
        return {
            "actionability_score": 0,
            "level": "EMPTY",
            "coverage_pct": 0,
            "layers_covered": 0,
            "total_layers": 6,
            "layer_breakdown": [],
            "gap_findings": [{"type": "EmptyInput", "severity": "HIGH", "message": "No review text provided."}],
            "is_actionable": False
        }

    clean = text.lower().strip()
    score = 0
    layers_status = []
    gaps = []

    # Layer 1: Subsystem / Component (Weight: 25)
    has_subsystem = bool(aspects) or any(
        kw in clean for kw in [
            "battery", "charging", "charger", "camera", "lens", "sensor", "photo", "display",
            "screen", "oled", "amoled", "refresh rate", "processor", "chip", "exynos", "snapdragon",
            "performance", "ram", "memory", "speaker", "audio", "mic", "bluetooth", "hinge", "keyboard"
        ]
    )
    if has_subsystem:
        score += 25
        layers_status.append({"layer": "Subsystem / Component", "covered": True, "weight": 25, "evidence": "Explicit component tagged"})
    else:
        layers_status.append({"layer": "Subsystem / Component", "covered": False, "weight": 25, "evidence": "No specific hardware/software part named"})
        gaps.append({
            "type": "MissingSubsystem",
            "severity": "HIGH",
            "message": "No specific hardware subsystem or component identified. Engineer cannot route ticket."
        })

    # Layer 2: Defect / Observable Failure Symptom (Weight: 25)
    defect_terms = [
        "overheat", "overheats", "overheating", "hot", "drain", "drains", "lag", "lags", "stutter",
        "stutters", "crash", "crashes", "freeze", "freezes", "slow", "blurry", "grainy", "noise",
        "dead pixel", "crack", "disconnect", "disconnects", "distorted", "unresponsive", "fails"
    ]
    praise_terms = [
        "crisp", "sharp", "snappy", "fast", "fluid", "smooth", "butter", "lasts", "punchy",
        "clear", "vibrant", "durable", "sturdy", "flawless", "superb"
    ]
    has_defect_or_symptom = any(t in clean for t in (defect_terms + praise_terms))
    if has_defect_or_symptom:
        score += 25
        layers_status.append({"layer": "Defect / Failure Symptom", "covered": True, "weight": 25, "evidence": "Concrete observable symptom found"})
    else:
        layers_status.append({"layer": "Defect / Failure Symptom", "covered": False, "weight": 25, "evidence": "Vague feedback without measurable symptom"})
        gaps.append({
            "type": "MissingSymptom",
            "severity": "HIGH",
            "message": "Review expresses generic sentiment without concrete observable symptom or measurement."
        })

    # Layer 3: Trigger & Usage Context (Weight: 20)
    has_trigger = any(re.search(r"\b" + re.escape(trig) + r"\b", clean) for trig in CONTEXT_TRIGGERS)
    if has_trigger:
        score += 20
        layers_status.append({"layer": "Trigger & Usage Context", "covered": True, "weight": 20, "evidence": "Environmental or operational context described"})
    else:
        layers_status.append({"layer": "Trigger & Usage Context", "covered": False, "weight": 20, "evidence": "Missing usage conditions"})
        gaps.append({
            "type": "MissingTrigger",
            "severity": "MEDIUM",
            "message": "Trigger context is unspecified. How was the device being operated when the issue occurred?"
        })

    # Layer 4: Polarity & Sentiment Valence (Weight: 10)
    if sentiment_compound is not None:
        has_clear_polarity = abs(sentiment_compound) >= 0.15
    else:
        has_clear_polarity = len(clean.split()) >= 4
    if has_clear_polarity:
        score += 10
        layers_status.append({"layer": "Polarity & Sentiment Valence", "covered": True, "weight": 10, "evidence": "Unambiguous polarity stance"})
    else:
        layers_status.append({"layer": "Polarity & Sentiment Valence", "covered": False, "weight": 10, "evidence": "Ambiguous or neutral stance"})
        gaps.append({
            "type": "UnresolvedPolarity",
            "severity": "LOW",
            "message": "Sentiment polarity is weak or ambiguous."
        })

    # Layer 5: Reproducibility / Frequency (Weight: 10)
    has_reproducibility = any(re.search(r"\b" + re.escape(rep) + r"\b", clean) for rep in REPRODUCIBILITY_MARKERS) or bool(re.search(r"\d+\s*(hours|hrs|minutes|mins|days|percent|%)", clean))
    if has_reproducibility:
        score += 10
        layers_status.append({"layer": "Reproducibility / Frequency", "covered": True, "weight": 10, "evidence": "Frequency or numeric time measurement included"})
    else:
        layers_status.append({"layer": "Reproducibility / Frequency", "covered": False, "weight": 10, "evidence": "No frequency or numeric time metric provided"})
        gaps.append({
            "type": "MissingFrequency",
            "severity": "LOW",
            "message": "No reproducibility indicator or numeric metric (e.g. 'drains in 4 hours', 'every time')."
        })

    # Layer 6: Expected Outcome / Benchmark (Weight: 10)
    has_expectation = any(re.search(r"\b" + re.escape(exp) + r"\b", clean) for exp in EXPECTATION_MARKERS)
    if has_expectation:
        score += 10
        layers_status.append({"layer": "Expected Outcome / Benchmark", "covered": True, "weight": 10, "evidence": "Target expectation or baseline comparison stated"})
    else:
        layers_status.append({"layer": "Expected Outcome / Benchmark", "covered": False, "weight": 10, "evidence": "No expected outcome baseline"})

    covered_count = sum(1 for l in layers_status if l["covered"])
    coverage_pct = round((covered_count / 6.0) * 100)

    if score >= 75:
        level = "HIGH"
        level_label = "Production-Ready Bug / Feature Feedback"
    elif score >= 50:
        level = "MEDIUM"
        level_label = "Moderately Actionable (Partial Context Missing)"
    else:
        level = "LOW"
        level_label = "Low Actionability (Subjective or Vague Vibe Check)"

    return {
        "actionability_score": score,
        "level": level,
        "level_label": level_label,
        "coverage_pct": coverage_pct,
        "layers_covered": covered_count,
        "total_layers": 6,
        "layer_breakdown": layers_status,
        "gap_findings": gaps,
        "is_actionable": score >= 50
    }


def synthesize_engineering_action_brief(
    product_name: str,
    aspect_name: str,
    nss_score: float,
    verified_quotes: List[str],
    category: str = "Smartphones",
    brand: str = "OEM"
) -> Dict[str, Any]:
    """
    Synthesizes a battle-tested 6-Section Engineering Action Brief (PRD / Jira Ticket)
    complete with Given/When/Then (BDD) Acceptance Criteria from analyzed customer feedback.
    """
    subsystem = SUBSYSTEM_MAP.get(aspect_name, f"{aspect_name} Subsystem Controller")
    primary_quote = verified_quotes[0] if verified_quotes else "Customer complaints indicate degraded user experience."

    # Identify primary symptom & trigger from quotes
    joined_text = " ".join(verified_quotes).lower()

    if "overheat" in joined_text or "charging" in joined_text:
        symptom = "Excessive thermal rise and slow charging ramp during high-amperage state."
        trigger = "Device is connected to 25W+ USB-PD charger in ambient room temperature (25°C+)."
        root_cause_hypothesis = "Aggressive charging curve with insufficient thermal dissipation throttling threshold."
        remediation = "Tune thermal hysteresis limits in thermal-engine.conf; enforce gradual stepped charging ramp above 38°C."
        bdd_given = "the device battery level is below 20% and ambient temperature is 25°C"
        bdd_when = "the user connects the official fast charger"
        bdd_then_1 = "charging temperature must remain below 41°C across all charging phases"
        bdd_then_2 = "charging current must throttle smoothly without complete thermal shutdown"
    elif "drain" in joined_text or "battery" in joined_text:
        symptom = "Rapid battery discharge rate under active network and display operations."
        trigger = "Continuous operation with 5G cellular connectivity, high brightness, or background media synchronization."
        root_cause_hypothesis = "Background modem wake-locks or unoptimized modem sleep states (RRC connection release)."
        remediation = "Audit modem RRC wake-lock timeouts; implement adaptive refresh rate clamp to 60Hz when battery < 20%."
        bdd_given = "the device is operating on 5G network with screen brightness at 75%"
        bdd_when = "the user engages in continuous navigation or video playback for 60 minutes"
        bdd_then_1 = "total battery discharge must not exceed 12% per hour"
        bdd_then_2 = "no runaway background wake-locks should be recorded in battery historian"
    elif "camera" in joined_text or "noise" in joined_text or "grain" in joined_text or "light" in joined_text:
        symptom = "Excessive digital noise and slow shutter latency in low-light capture."
        trigger = "Camera capture in scenes with illuminance under 10 Lux using primary wide sensor."
        root_cause_hypothesis = "Aggressive multi-frame ISO noise reduction blurring fine textures and high exposure shutter lag."
        remediation = "Retrain raw multi-frame HDR/Night mode neural network weights; optimize zero-shutter-lag (ZSL) buffer depth."
        bdd_given = "the camera app is active in a low-light indoor environment (< 10 Lux)"
        bdd_when = "the user taps the shutter button to capture a portrait or moving subject"
        bdd_then_1 = "shutter capture latency must not exceed 180 ms"
        bdd_then_2 = "signal-to-noise ratio (SNR) must meet ISO 12232 standard with zero motion blur artifacts"
    elif "lag" in joined_text or "stutter" in joined_text or "slow" in joined_text:
        symptom = "UI frame drops, micro-stutters, and sluggish app switching under everyday workloads."
        trigger = "Switching between heavy memory-consuming applications or running GPS navigation."
        root_cause_hypothesis = "SoC DVFS governor thermal throttling too eagerly or aggressive background RAM compression."
        remediation = "Re-calibrate interactive governor boost frequencies; optimize zRAM swap threshold to avoid page thrashing."
        bdd_given = "the device has 5 background applications open and 1 active navigation app"
        bdd_when = "the user performs rapid gesture app-switching or UI navigation"
        bdd_then_1 = "99th percentile UI frame rate must remain above 58 fps (on 60Hz) or 115 fps (on 120Hz)"
        bdd_then_2 = "zero jank or ANR (Application Not Responding) events should be logged"
    else:
        symptom = f"Inconsistent user experience flagged in {aspect_name} feedback."
        trigger = "Standard everyday consumer operating conditions."
        root_cause_hypothesis = f"Sub-optimal calibration in {subsystem} logic."
        remediation = f"Review telemetry data and run comprehensive QA regression suite on {subsystem}."
        bdd_given = f"the {product_name} is running in certified operating conditions"
        bdd_when = f"the user exercises {aspect_name} capabilities"
        bdd_then_1 = f"operational performance must meet OEM reference specifications"
        bdd_then_2 = f"negative user sentiment must decline below 10% in subsequent telemetry"

    ticket_id = f"RL-{brand.upper()[:3]}-{re.sub(r'[^A-Za-z0-9]', '', aspect_name).upper()[:4]}-001"

    # Assemble structured 6-section brief
    sections = {
        "01_context": {
            "title": "01 · Issue Context & Review Impact",
            "ticket_id": ticket_id,
            "product": product_name,
            "category": category,
            "aspect": aspect_name,
            "net_sentiment_score": f"{nss_score:+.1f} NSS",
            "priority": "P1 - High Priority" if nss_score < -40 else ("P2 - Medium Priority" if nss_score < 0 else "P3 - Feature Improvement"),
            "summary": f"{aspect_name} feedback on {product_name} exhibits significant customer dissatisfaction ({nss_score:+.1f} NSS), representing a primary purchase trade-off."
        },
        "02_subsystem": {
            "title": "02 · Affected Architecture & Subsystem",
            "component": subsystem,
            "firmware_module": f"sys.{aspect_name.lower().replace(' ', '_')}.daemon",
            "impact_scope": "Core Customer Experience & Return Rate Driver"
        },
        "03_evidence": {
            "title": "03 · Verified Customer Review Evidence",
            "primary_quote": primary_quote,
            "all_quotes": verified_quotes[:4],
            "data_provenance": "ReviewLens Verified Feedback Extraction"
        },
        "04_reproduction": {
            "title": "04 · Reproduction Journey & Trigger Conditions",
            "symptom": symptom,
            "trigger_condition": trigger,
            "root_cause_hypothesis": root_cause_hypothesis
        },
        "05_remediation": {
            "title": "05 · Proposed Engineering Remediation",
            "action_plan": remediation,
            "verification_method": "Automated Hardware-in-the-Loop (HIL) & Stress Telemetry Test"
        },
        "06_acceptance_criteria": {
            "title": "06 · BDD Acceptance Criteria (Given / When / Then)",
            "scenario": f"Verify {aspect_name} reliability under stress conditions",
            "given": bdd_given,
            "when": bdd_when,
            "then": [bdd_then_1, bdd_then_2]
        }
    }

    return sections


def export_brief_to_jira_markdown(brief: Dict[str, Any]) -> str:
    """
    Formats the 6-Section Engineering Action Brief as a publication-ready Jira / Linear issue.
    """
    c = brief["01_context"]
    sub = brief["02_subsystem"]
    ev = brief["03_evidence"]
    rp = brief["04_reproduction"]
    rem = brief["05_remediation"]
    ac = brief["06_acceptance_criteria"]

    lines = [
        f"# [{c['ticket_id']}] {c['product']} — Customer Feedback Bug: {c['aspect']} ({c['net_sentiment_score']})",
        "",
        f"**Priority:** `{c['priority']}` | **Component:** `{sub['component']}` | **Module:** `{sub['firmware_module']}`",
        "",
        "---",
        "",
        "### 01 · Issue Context & Impact",
        f"- **Product:** {c['product']} ({c['category']})",
        f"- **Aspect Dimension:** {c['aspect']}",
        f"- **Net Sentiment Score (NSS):** `{c['net_sentiment_score']}`",
        f"- **Executive Summary:** {c['summary']}",
        "",
        "### 02 · Affected Architecture & Subsystem",
        f"- **Target Component:** `{sub['component']}`",
        f"- **Firmware Subsystem:** `{sub['firmware_module']}`",
        f"- **Impact Scope:** {sub['impact_scope']}",
        "",
        "### 03 · Verified Customer Review Evidence",
        "Direct verbatim customer complaint extracts cited from ReviewLens corpus:",
    ]

    for q in ev["all_quotes"]:
        lines.append(f"> *\"{q}\"*")

    lines.extend([
        "",
        "### 04 · Reproduction Journey & Trigger Conditions",
        f"- **Observed Symptom:** {rp['symptom']}",
        f"- **Trigger Environment:** {rp['trigger_condition']}",
        f"- **Root Cause Hypothesis:** {rp['root_cause_hypothesis']}",
        "",
        "### 05 · Proposed Engineering Remediation",
        f"- **Action Plan:** {rem['action_plan']}",
        f"- **Validation Test:** {rem['verification_method']}",
        "",
        "### 06 · BDD Acceptance Criteria (Given / When / Then)",
        "```gherkin",
        f"Scenario: {ac['scenario']}",
        f"  Given {ac['given']}",
        f"  When {ac['when']}",
    ])

    for t in ac["then"]:
        lines.append(f"  Then {t}")

    lines.extend([
        "```",
        "",
        "---",
        f"*Generated automatically by ReviewLens Requirement Engineering Engine (Vector-Inspired Specification Compiler).*"
    ])

    return "\n".join(lines)
