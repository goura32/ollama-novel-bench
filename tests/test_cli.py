import pytest

from novelbench.cli import build_parser, select_run_models


@pytest.fixture
def profiles():
    return [
        {
            "name": "first",
            "capabilities": ["completion", "thinking"],
            "thinking": {"modes": []},
        },
        {"name": "second", "capabilities": ["completion"], "thinking": {"modes": []}},
    ]


def test_smoke_defaults_to_one_deterministic_model(profiles):
    selected = select_run_models(profiles, None, "smoke")
    assert [model["name"] for model in selected] == ["first"]


def test_smoke_rejects_multiple_explicit_models(profiles):
    with pytest.raises(ValueError, match="exactly one"):
        select_run_models(profiles, "first,second", "smoke")


def test_non_smoke_keeps_all_models_without_filter(profiles):
    selected = select_run_models(profiles, None, "quick")
    assert [model["name"] for model in selected] == ["first", "second"]


def test_cli_separates_local_models_from_cloud_reference_flags():
    parser = build_parser()
    default = parser.parse_args(["run", "--profile", "quick", "--run-id", "q1"])
    assert default.models is None
    assert default.cloud_reference is False
    assert default.no_cloud_reference is False
    assert default.cloud_reference_only is False

    explicit = parser.parse_args(
        [
            "run",
            "--profile",
            "smoke",
            "--run-id",
            "s1",
            "--models",
            "local-a,local-b",
            "--cloud-reference",
        ]
    )
    assert explicit.models == "local-a,local-b"
    assert explicit.cloud_reference is True


def test_cli_accepts_explicit_cloud_reference_only_smoke():
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--profile", "smoke", "--run-id", "cloud-smoke", "--cloud-reference-only"]
    )
    assert args.cloud_reference_only is True


def test_audit_luna_is_idempotent_by_default_and_force_is_explicit():
    parser = build_parser()
    normal = parser.parse_args(["audit-luna", "run-1"])
    forced = parser.parse_args(["audit-luna", "run-1", "--force"])
    assert normal.force is False
    assert forced.force is True
