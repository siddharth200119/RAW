"""The base install (no extras) must be able to import RAW, RAW.models and RAW.utils.

Runs in a subprocess with the optional dependencies blocked, so a top-level
import of numpy/cv2/boto3/tiktoken in a base module fails here.
"""
import subprocess
import sys
import textwrap

OPTIONAL_DEPS = ["numpy", "cv2", "boto3", "botocore", "tiktoken"]


def run_without_extras(code: str) -> subprocess.CompletedProcess:
    blocker = textwrap.dedent(f"""
        import sys
        BLOCKED = {OPTIONAL_DEPS!r}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked optional dependency: {{name}}")
                return None

        sys.meta_path.insert(0, Blocker())
    """)
    return subprocess.run(
        [sys.executable, "-c", blocker + textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_base_modules_import_without_extras():
    result = run_without_extras("""
        import io, tempfile
        import RAW
        from RAW import Message, Tool, LLMCapability, Metrics, NullMetrics
        from RAW.models import File, FileType, Skill, jsonschema, Image
        from RAW.utils import Logger, RequestsClient, Router, count_tokens
        from RAW.utils.storage import Storage, LocalStorage, S3Storage
        from RAW.deciders import BaseDecider, OpenRouterDecider

        storage = LocalStorage(root=tempfile.mkdtemp())
        f = File(name="a.txt", file_type=FileType.TEXT, storage=storage, location="a.txt")
        storage.save(f, io.BytesIO(b"hi"))
        assert f.open().read() == b"hi"
        assert count_tokens("12345678") == 2
    """)
    assert result.returncode == 0, result.stderr


def test_extras_raise_install_hint_without_extras():
    result = run_without_extras("""
        import RAW
        from RAW.models import Image
        from RAW.utils.storage import S3Storage

        for name, fn, extra in [
            ("Image", lambda: Image.from_bytes(b"x"), "agentic"),
            ("S3Storage", lambda: S3Storage(client=None), "s3"),
            ("Agent", lambda: RAW.Agent, "agentic"),
        ]:
            try:
                fn()
            except ImportError as e:
                assert f"raw-agents[{extra}]" in str(e), (name, e)
            else:
                raise AssertionError(f"{name} worked without the {extra} extra")
    """)
    assert result.returncode == 0, result.stderr
