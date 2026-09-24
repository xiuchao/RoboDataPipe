from __future__ import annotations

import unittest

from vlm.task_dashboard import records_from_summary, task_outcome_dashboard_html


class TaskDashboardTest(unittest.TestCase):
    def test_summary_parser_and_dashboard_preserve_task_gate_semantics(self) -> None:
        summary = """[episode] ep000
[episode_inference_seconds] 1.100
```json
{"placement_status": "inside", "confidence": 0.95}
```
[episode] ep001
[episode_inference_seconds] 1.200
```json
{"placement_status": "uncertain", "confidence": 0.5}
```
[episode] ep002
[episode_inference_seconds] 1.300
not valid JSON
"""

        records = records_from_summary(summary)

        self.assertEqual(
            [(record.episode_id, record.status) for record in records],
            [("ep000", "inside"), ("ep001", "uncertain"), ("ep002", "unparsed")],
        )
        self.assertEqual(records[1].confidence, 0.5)
        html = task_outcome_dashboard_html(records, "demo")
        self.assertIn("33.3%", html)
        self.assertIn("ep001", html)
        self.assertIn("Review queue", html)
        self.assertIn("Export review JSON", html)
        self.assertIn("localStorage.setItem", html)
        self.assertIn("reviewed_at", html)
        self.assertIn("Plotly.newPlot", html)


if __name__ == "__main__":
    unittest.main()