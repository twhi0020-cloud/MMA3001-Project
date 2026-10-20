"""The README stays true: numbers match the results tables, links and commands point at real things.

A README that quotes a result, or names a script, goes wrong the moment the code or results change.
These tests compare it with the files it describes.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _observed():
    t = pd.read_csv(ROOT / "results" / "final_test_main.csv")
    return t[t["subset"] == "observed"]


def test_no_unfinished_markers():
    assert not re.search(r"\bTODO\b|\bTBD\b|FIXME", README)


def test_has_the_sections_the_brief_asks_a_project_to_explain():
    for heading in ("Engineering problem", "Inputs and outputs", "Approach", "Results", "Known data issues",
                    "Limitations", "Getting started", "Repository layout", "AI use", "Licence"):
        assert re.search(rf"^#+ .*{heading}", README, flags=re.MULTILINE), f"README has no '{heading}' section"


def _slug(heading):
    """GitHub's anchor for a heading: lower case, punctuation removed, spaces to hyphens."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def test_relative_links_and_contents_anchors_point_at_real_targets():
    links = re.findall(r"\]\(([^)\s]+)\)", README)
    local = [l.split("#")[0] for l in links if not l.startswith(("http://", "https://", "mailto:", "#"))]
    assert local, "README has no relative links"
    missing = [l for l in local if l and not (ROOT / l).exists()]
    assert not missing, f"README links to files that do not exist: {missing}"
    anchors = {_slug(h) for h in re.findall(r"^#+\s+(.*)$", README, flags=re.MULTILINE)}
    jumps = [l[1:] for l in links if l.startswith("#")]
    assert jumps, "README has no contents links"
    broken = [a for a in jumps if a not in anchors]
    assert not broken, f"contents links that match no heading: {broken}"


def test_scripts_named_in_commands_exist_and_options_are_real():
    cmds = re.findall(r"python scripts/(\w+\.py)([^\n`]*)", README)
    assert cmds, "README shows no script commands"
    for name, rest in cmds:
        path = ROOT / "scripts" / name
        assert path.exists(), f"README runs scripts/{name}, which does not exist"
        for opt in re.findall(r"(--[a-z][a-z-]*)", rest):
            assert opt in path.read_text(encoding="utf-8"), f"README uses {opt} with scripts/{name}, which does not define it"


def test_headline_test_results_match_the_results_table():
    o = _observed()
    for h in (15, 30, 60):
        d = o[o["horizon"] == h].set_index("method")
        assert f"{d.loc['persistence', 'rmse']:.2f}" in README
        for m in ("decision_tree", "spline_hybrid", "linear_regression"):
            r = d.loc[m]
            cell = f"{r['improvement_pct']:.1f}% ({r['improvement_lo']:.1f} to {r['improvement_hi']:.1f})"
            assert cell in README, f"README does not quote {m} at {h} min as '{cell}'"


def test_data_sizes_and_chosen_tree_match_the_results_tables():
    s = pd.read_csv(ROOT / "results" / "final_test_sizes.csv").set_index("horizon").loc[30]
    for key in ("train_rows", "val_rows", "test_rows"):
        assert f"{int(s[key]):,}" in README, f"README does not quote {key} = {int(s[key]):,}"
    assert f"depth {int(s['max_depth'])}" in README and f"{int(s['n_leaves'])} leaves" in README


def test_test_count_and_coverage_match_the_reports():
    root = ET.parse(ROOT / "reports" / "junit.xml").getroot()
    suite = root if root.tag == "testsuite" else root[0]
    assert f"{suite.attrib['tests']} tests" in README, f"README should say '{suite.attrib['tests']} tests' (from reports/junit.xml)"
    total = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+%)", (ROOT / "reports" / "coverage.txt").read_text()).group(1)
    assert f"{total} line coverage" in README, f"README should say '{total} line coverage' (from reports/coverage.txt)"


def test_live_function_and_speedup_figures_match_the_results_tables():
    lv = pd.read_csv(ROOT / "results" / "live_vs_batch.csv").set_index("metric")["lookback_24h"]
    assert f"{lv['live_forecast_available_%']:.1f}%" in README and f"{lv['identical_to_batch_%_of_available']:.1f}%" in README
    ts = pd.read_csv(ROOT / "results" / "tuning_strategies.csv").set_index("strategy")
    assert f"{ts.loc['fast', 'speedup_vs_exhaustive']:.1f}x" in README
