import json

from novelbench.judge import build_pairwise_prompt, parse_pairwise_json


def test_pairwise_prompt_has_both_orders_without_target_metadata():
    prompt = build_pairwise_prompt(
        task="静かな駅の場面を書く",
        work_a="Aの本文",
        work_b="Bの本文",
        order="A_B",
    )
    assert "<BEGIN_A>" in prompt and "<BEGIN_B>" in prompt
    assert "Aの本文" in prompt and "Bの本文" in prompt
    assert "model" not in prompt.lower()

    parsed = parse_pairwise_json(
        json.dumps({"winner": "A", "margin": 2, "brief_rationale": "Aの方が自然"}),
    )
    assert parsed["winner"] == "A"
