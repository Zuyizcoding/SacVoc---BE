"""Build/verify the backend with isolated containers and a disposable SQLite volume.

This is a local integration check, not deployment. Only resources created by this
run are removed, by the exact IDs Docker returned. Existing Compose services,
user volumes and images are never pruned. No Gemini request is made.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-8000:]}")
    return result.stdout.strip()


def _get_json(url: str, *, headers: dict | None = None) -> dict:
    with urlopen(Request(url, headers=headers or {}), timeout=2) as response:
        return json.load(response)


def _wait_for_health(container_id: str) -> tuple[str, dict]:
    address = docker("port", container_id, "8000/tcp").splitlines()[0]
    base_url = "http://" + address
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            health = _get_json(base_url + "/health")
            if health.get("status") == "ok":
                return base_url, health
        except (URLError, OSError):
            pass
        time.sleep(0.25)
    raise RuntimeError(
        "Container health failed: " + docker("logs", container_id, check=False)
    )


def verify_runtime(image: str, *, build: bool = False) -> dict:
    if docker("info", "--format", "{{.OSType}}") != "linux":
        raise ValueError("A Linux Docker Engine is required")
    if build:
        docker("build", "--tag", image, ".")
    image_info = json.loads(docker("image", "inspect", image))[0]
    if image_info["Config"]["User"] != "10001:10001":
        raise ValueError("Runtime image must use the dedicated non-root user")
    token = uuid4().hex
    volume = "sacvoc-check-" + token
    docker("volume", "create", "--label", "sacvoc.verification=true", volume)
    containers = []

    def create_container(suffix: str) -> str:
        ident = docker(
            "create",
            "--name",
            "sacvoc-check-" + token + "-" + suffix,
            "--label",
            "sacvoc.verification=true",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/sacvoc",
            "--publish",
            "127.0.0.1::8000",
            "--env",
            "GEMINI_API_KEY=",
            "--env",
            "DEBUG=false",
            image,
        )
        containers.append(ident)
        docker("start", ident)
        return ident

    try:
        first = create_container("first")
        base_url, health = _wait_for_health(first)
        history = _get_json(
            base_url + "/api/v1/analyze/history", headers={"X-Device-ID": "a" * 64}
        )
        if history["data"] != {"color": None, "body": None}:
            raise ValueError("A fresh temporary volume must have empty history")
        ready = _get_json(base_url + "/health/ready")
        if ready["status"] != "degraded":
            raise ValueError("An unseeded catalog must report degraded readiness")
        docker(
            "exec",
            first,
            "python",
            "-c",
            "import importlib.util,os,pathlib; "
            "assert os.getuid()==10001; "
            "assert importlib.util.find_spec('torch') is None; "
            "assert importlib.util.find_spec('pytest') is None; "
            "assert not pathlib.Path('/app/.env').exists(); "
            "assert not pathlib.Path('/app/.git').exists()",
        )
        docker(
            "exec",
            first,
            "python",
            "-c",
            "import sqlite3; "
            "db=sqlite3.connect('/var/lib/sacvoc/sac_voc.db'); "
            "db.execute('CREATE TABLE docker_smoke_marker(value TEXT)'); "
            "db.execute(\"INSERT INTO docker_smoke_marker VALUES ('persisted')\"); "
            "db.commit(); db.close()",
        )
        docker("rm", "--force", first)
        containers.remove(first)
        second = create_container("replacement")
        _wait_for_health(second)
        persisted = docker(
            "exec",
            second,
            "python",
            "-c",
            "import sqlite3; "
            "db=sqlite3.connect('/var/lib/sacvoc/sac_voc.db'); "
            "print(db.execute('SELECT value FROM docker_smoke_marker').fetchone()[0]); db.close()",
        )
        if persisted != "persisted":
            raise ValueError("SQLite data did not persist across container replacement")
        return {
            "image": image,
            "image_id": image_info["Id"],
            "docker_server_version": docker(
                "version", "--format", "{{.Server.Version}}"
            ),
            "python_version": docker("exec", second, "python", "--version"),
            "health": health,
            "fresh_history": history["data"],
            "empty_catalog_readiness": ready["status"],
            "runtime_uid": 10001,
            "sqlite_persisted_after_container_replacement": True,
            "gemini_requests": 0,
        }
    finally:
        for ident in containers:
            docker("rm", "--force", ident, check=False)
        docker("volume", "rm", volume, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="sacvoc-backend:local")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new verification output path")
    report = verify_runtime(args.image, build=args.build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Container integration checks passed: {args.output}")


if __name__ == "__main__":
    main()
