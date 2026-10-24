"""
Actionable Recommendation Engine for ReviewLens.
Translates negative sentiment patterns and customer pain points into prioritized engineering & PM actions.
"""

from typing import List, Dict, Any
from reviewlens.models import AspectSummary


class ActionRecommender:
    """Generates prioritized tactical actions to fix product vulnerabilities."""

    ACTION_PLAYBOOK = {
        "Battery & Power": {
            "title": "Optimize Battery Management & Power Delivery",
            "action": "Investigate standby drain via firmware update and test charging circuitry for heating or slow charging cycles.",
            "team": "Firmware & Hardware Engineering",
        },
        "Sound & Audio": {
            "title": "Acoustic Tuning & Microphone Calibration",
            "action": "Rebalance EQ profile to address muffled mid-ranges or harsh treble; improve noise suppression algorithms for microphone clarity.",
            "team": "Acoustics & DSP Engineering",
        },
        "Build Quality & Durability": {
            "title": "Material & Structural Reinforcement",
            "action": "Audit structural stress points (hinges, clips, casing) and switch away from brittle plastics to reinforced composite or metal alloys.",
            "team": "Mechanical & Industrial Design",
        },
        "Comfort & Ergonomics": {
            "title": "Ergonomic & Cushioning Redesign",
            "action": "Reduce clamping pressure, increase memory foam density on contact pads, and evaluate weight distribution for long wear sessions.",
            "team": "Industrial Design & User Research",
        },
        "Display & Visuals": {
            "title": "Display Brightness & Color Calibration",
            "action": "Increase outdoor peak nit brightness, improve anti-reflective coating, and calibrate factory color profiles.",
            "team": "Display Engineering",
        },
        "Performance & Speed": {
            "title": "Software Optimization & Latency Reduction",
            "action": "Profile memory leaks, remove bloatware background processes, and optimize animation framerates to eliminate UI stutter.",
            "team": "Software & QA Team",
        },
        "Price & Value for Money": {
            "title": "Reposition Value Proposition or Bundle Accessories",
            "action": "Address perceived price mismatch by bundling high-value accessories (cables, cases) or introducing tiered promotional pricing.",
            "team": "Product Marketing & Commercial Strategy",
        },
        "Ease of Use & Connectivity": {
            "title": "Streamline Onboarding & Bluetooth Stability",
            "action": "Resolve pairing drops by updating Bluetooth stack firmware and redesigning companion mobile app onboarding steps.",
            "team": "Software & QA Team",
        },
        "Customer Support & Warranty": {
            "title": "Enhance RMA Turnaround & Support Training",
            "action": "Retrain customer support agents on common warranty issues and streamline the return/replacement ticket resolution time.",
            "team": "Customer Experience & Operations",
        },
        "Packaging & Delivery": {
            "title": "Upgrade In-Transit Protection Packaging",
            "action": "Enhance inner box buffer padding and seal protection to eliminate reports of cosmetic damage during courier transit.",
            "team": "Supply Chain & Packaging",
        },
    }

    def generate_recommendations(self, aspect_summaries: List[AspectSummary]) -> List[Dict[str, Any]]:
        """
        Produce prioritized recommendations sorted by negative impact severity.
        """
        recommendations = []

        # Sort aspects by negative mentions count descending
        problematic_aspects = [a for a in aspect_summaries if a.negative_mentions > 0 or a.net_sentiment_score < 10]
        problematic_aspects.sort(key=lambda a: (a.negative_mentions * (100 - a.net_sentiment_score)), reverse=True)

        for aspect in problematic_aspects:
            playbook_entry = self.ACTION_PLAYBOOK.get(
                aspect.category,
                {
                    "title": f"Address {aspect.category} Deficiencies",
                    "action": f"Conduct focused qualitative audit of {aspect.category} complaints and implement corrective revisions.",
                    "team": "Product Management",
                }
            )

            # Determine severity
            neg_ratio = aspect.sentiment_ratio.get("negative", 0.0)
            if aspect.net_sentiment_score < -20 or neg_ratio > 40:
                priority = "CRITICAL"
                badge_color = "red"
            elif aspect.net_sentiment_score < 15 or neg_ratio > 20:
                priority = "HIGH"
                badge_color = "orange"
            else:
                priority = "MEDIUM"
                badge_color = "yellow"

            recommendations.append({
                "category": aspect.category,
                "priority": priority,
                "badge_color": badge_color,
                "title": playbook_entry["title"],
                "recommended_action": playbook_entry["action"],
                "responsible_team": playbook_entry["team"],
                "negative_mentions": aspect.negative_mentions,
                "net_sentiment_score": aspect.net_sentiment_score,
                "sample_complaint": aspect.sample_negative_quotes[0] if aspect.sample_negative_quotes else "N/A"
            })

        return recommendations
