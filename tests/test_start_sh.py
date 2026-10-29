import subprocess
import textwrap
from pathlib import Path

import pytest

START_SH = Path(__file__).resolve().parent.parent / "start.sh"

# Extract just check_port_registry() from start.sh and run it against a
# caller-supplied FUSION_REGISTRY / PORT, without ever starting the service.
_RUNNER = textwrap.dedent(
    """\
    source /dev/stdin <<<"$(sed -n '/^check_port_registry/,/^}/p' "$1")"
    check_port_registry
    """
)


def _run_check(registry: str, port: str = "11444", extra_env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "FUSION_REGISTRY": registry, "PORT": port}
    for k, v in (extra_env or {}).items():
        env[k] = v
    return subprocess.run(
        ["bash", "-c", _RUNNER, "runner", str(START_SH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def _registry_with_entry(port: int = 11444) -> str:
    return textwrap.dedent(
        f"""\
        services:
          fusion-model-hub:
            port: {port}
            purpose: test
        """
    )


@pytest.fixture
def reg_ok(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(_registry_with_entry())
    return str(p)


@pytest.fixture
def reg_missing_entry(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text("services:\n  fusion-mlx:\n    port: 11434\n")
    return str(p)


class TestStartShRegistryCheck:
    def test_passes_when_port_registered(self, reg_ok):
        result = _run_check(reg_ok)
        assert result.returncode == 0, result.stderr

    def test_refuses_when_entry_missing(self, reg_missing_entry):
        result = _run_check(reg_missing_entry)
        assert result.returncode == 1
        assert "not registered" in result.stderr
        assert "refusing to start" in result.stderr

    def test_refuses_when_port_mismatch(self, tmp_path):
        p = tmp_path / "registry.yaml"
        p.write_text(_registry_with_entry(port=11999))
        result = _run_check(str(p))
        assert result.returncode == 1
        assert "not registered" in result.stderr

    def test_bypass_env_skips_refusal(self, reg_missing_entry):
        result = _run_check(reg_missing_entry, extra_env={"FUSION_SKIP_REGISTRY_CHECK": "1"})
        assert result.returncode == 0
        assert "bypassed" in result.stderr

    def test_missing_registry_file_warns_and_continues(self, tmp_path):
        result = _run_check(str(tmp_path / "nope.yaml"))
        assert result.returncode == 0
        assert "self-check skipped" in result.stderr

    def test_real_workspace_registry_passes(self):
        registry = Path.home() / "fusion" / "architecture" / "port-registry.yaml"
        if not registry.is_file():
            pytest.skip("workspace port-registry.yaml not present")
        result = _run_check(str(registry))
        assert result.returncode == 0, result.stderr


class TestStartShLegacyKeyNames:
    def test_no_legacy_key_env_names_in_start_sh(self):
        # #74: canonical FUSION_CLIENT_KEY only — the legacy alias chain lives
        # in fusion_core.config. check-env rule 2c escalates to error 2027-01-01.
        content = START_SH.read_text()
        for legacy in ("FUSION_MLX_API_KEY", "MLX_API_KEY", "FUSION_API_KEY", "FUSION_GATEWAY_ADMIN_KEY"):
            assert legacy not in content, f"legacy env name {legacy} still referenced in start.sh"

    def test_no_legacy_key_env_names_in_server_config(self):
        content = (START_SH.parent / "fusion_model_hub" / "server" / "config.py").read_text()
        for legacy in ("FUSION_MLX_API_KEY", "MLX_API_KEY", "FUSION_API_KEY", "FUSION_GATEWAY_ADMIN_KEY"):
            assert legacy not in content, f"legacy env name {legacy} still referenced in config.py"
