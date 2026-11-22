"""
Unit tests for modules/action_brief.py
Vector-inspired requirement engineering, review actionability index (0-100),
6-section engineering briefs, and Given/When/Then (BDD) acceptance criteria synthesis.
"""

import unittest
from modules.action_brief import (
    evaluate_review_actionability,
    synthesize_engineering_action_brief,
    export_brief_to_jira_markdown,
    ACTIONABILITY_LAYERS,
    SUBSYSTEM_MAP
)
from modules.live_analyzer import analyze_custom_review


class TestActionBrief(unittest.TestCase):

    def test_evaluate_review_actionability_empty_input(self):
        """Empty or whitespace input should return score 0, level EMPTY, and EmptyInput gap."""
        res = evaluate_review_actionability("")
        self.assertEqual(res["actionability_score"], 0)
        self.assertEqual(res["level"], "EMPTY")
        self.assertFalse(res["is_actionable"])
        self.assertEqual(res["layers_covered"], 0)
        self.assertTrue(any(g["type"] == "EmptyInput" for g in res["gap_findings"]))

        res_none = evaluate_review_actionability(None)
        self.assertEqual(res_none["actionability_score"], 0)
        self.assertEqual(res_none["level"], "EMPTY")

    def test_evaluate_review_actionability_high_quality_feedback(self):
        """Rich feedback with component, trigger, reproducible metrics, and expectation scores HIGH."""
        feedback = (
            "The battery drains completely within 3 hours when recording 4k video at 60fps on 5G. "
            "It constantly overheats to 45C and the phone shuts down repeatedly. "
            "I expected at least 7 hours compared to my previous model."
        )
        res = evaluate_review_actionability(feedback, aspects=["battery", "camera", "cooling"], sentiment_compound=-0.85)

        self.assertGreaterEqual(res["actionability_score"], 75)
        self.assertEqual(res["level"], "HIGH")
        self.assertTrue(res["is_actionable"])
        self.assertGreaterEqual(res["layers_covered"], 5)
        self.assertEqual(res["total_layers"], 6)

        # Check layer breakdown
        covered_layers = [l["layer"] for l in res["layer_breakdown"] if l["covered"]]
        self.assertIn("Subsystem / Component", covered_layers)
        self.assertIn("Defect / Failure Symptom", covered_layers)
        self.assertIn("Trigger & Usage Context", covered_layers)

    def test_evaluate_review_actionability_vague_input_gaps(self):
        """Vague feedback without component or trigger produces explicit gap findings."""
        vague_text = "Bad product, hated it."
        res = evaluate_review_actionability(vague_text, aspects=[], sentiment_compound=-0.6)

        self.assertLess(res["actionability_score"], 50)
        self.assertEqual(res["level"], "LOW")
        self.assertFalse(res["is_actionable"])

        gap_types = [g["type"] for g in res["gap_findings"]]
        self.assertIn("MissingSubsystem", gap_types)
        self.assertIn("MissingTrigger", gap_types)

    def test_synthesize_engineering_action_brief_structure(self):
        """Synthesized brief must strictly contain all 6 architectural sections and valid BDD."""
        quotes = [
            "Charging speed is disappointing and the device overheats severely when plugged in with 45W charger."
        ]
        brief = synthesize_engineering_action_brief(
            product_name="Samsung Galaxy S24 FE",
            aspect_name="Charging",
            nss_score=-75.0,
            verified_quotes=quotes,
            category="Smartphones",
            brand="Samsung"
        )

        expected_sections = [
            "01_context",
            "02_subsystem",
            "03_evidence",
            "04_reproduction",
            "05_remediation",
            "06_acceptance_criteria"
        ]
        for sec in expected_sections:
            self.assertIn(sec, brief, f"Missing required brief section: {sec}")

        # Check Ticket ID convention
        c = brief["01_context"]
        self.assertTrue(c["ticket_id"].startswith("RL-SAM-"))
        self.assertEqual(c["product"], "Samsung Galaxy S24 FE")
        self.assertEqual(c["priority"], "P1 - High Priority")

        # Check Subsystem Mapping
        sub = brief["02_subsystem"]
        self.assertIn("Charging", sub["component"])

        # Check BDD Gherkin Acceptance Criteria
        ac = brief["06_acceptance_criteria"]
        self.assertIn("Scenario:", f"Scenario: {ac['scenario']}")
        self.assertTrue(len(ac["given"]) > 0)
        self.assertTrue(len(ac["when"]) > 0)
        self.assertEqual(len(ac["then"]), 2)

    def test_export_brief_to_jira_markdown(self):
        """Exported Jira Markdown must include headers, Gherkin code block, and quote citations."""
        brief = synthesize_engineering_action_brief(
            product_name="Sony WH-1000XM5",
            aspect_name="Sound quality",
            nss_score=-30.0,
            verified_quotes=["Treble is muffled and ANC introduces low-frequency hiss."],
            category="Headphones",
            brand="Sony"
        )
        md = export_brief_to_jira_markdown(brief)

        self.assertIn("# [RL-SON-SOUN-001]", md)
        self.assertIn("### 01 · Issue Context & Impact", md)
        self.assertIn("### 02 · Affected Architecture & Subsystem", md)
        self.assertIn("### 03 · Verified Customer Review Evidence", md)
        self.assertIn("### 04 · Reproduction Journey & Trigger Conditions", md)
        self.assertIn("### 05 · Proposed Engineering Remediation", md)
        self.assertIn("### 06 · BDD Acceptance Criteria (Given / When / Then)", md)
        self.assertIn("```gherkin", md)
        self.assertIn("Scenario:", md)
        self.assertIn("Given ", md)
        self.assertIn("When ", md)
        self.assertIn("Then ", md)

    def test_live_analyzer_actionability_integration(self):
        """Live analyzer pipeline must seamlessly attach actionability evaluations."""
        review = "Screen display is vibrant and sharp, but battery life drops by 50% in 2 hours on 5G."
        res = analyze_custom_review(review, rating=3.0, category="Smartphones")

        self.assertEqual(res["status"], "ok")
        self.assertIn("actionability", res)
        act = res["actionability"]
        self.assertIn("actionability_score", act)
        self.assertIn("level", act)
        self.assertIn("coverage_pct", act)
        self.assertIn("layer_breakdown", act)
        self.assertIn("gap_findings", act)
        self.assertGreater(act["actionability_score"], 0)


if __name__ == "__main__":
    unittest.main()
