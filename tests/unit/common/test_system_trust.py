"""SystemTrust.install() and the launcher's promise to call it before anything can build an SSL context."""
import ast
import ssl
import sys
from unittest.mock import patch

import pytest

from common import SystemTrust
from common.Paths import AISERVER_DIR


@pytest.fixture
def stock_ssl():
    """Put the interpreter's own SSLContext back afterwards: the swap is process-wide."""
    import truststore
    truststore.extract_from_ssl()
    yield
    truststore.extract_from_ssl()


@pytest.mark.unit
class TestInstall:
    def test_new_contexts_ask_the_system_trust_store(self, stock_ssl):
        import truststore
        assert not isinstance(ssl.create_default_context(), truststore.SSLContext)

        SystemTrust.install()

        assert isinstance(ssl.create_default_context(), truststore.SSLContext)

    def test_a_missing_package_leaves_the_stock_verification_in_place(self, stock_ssl):
        stock = ssl.SSLContext
        with patch.dict(sys.modules, {"truststore": None}):
            status = SystemTrust.install()

        assert ssl.SSLContext is stock
        assert "not installed" in status


@pytest.mark.unit
class TestLauncherOrder:
    def test_the_launcher_installs_it_before_importing_anything_that_could_build_a_context(self):
        # aiohttp builds its default SSL context at import: installed after that, the swap
        # would miss it. Only these may be imported ahead of the call.
        harmless = {"grpc_env", "sys", "pathlib", "common"}
        body = ast.parse((AISERVER_DIR / "floxy_ai_server.py").read_text()).body
        imported_before = set()
        for node in body:
            if isinstance(node, ast.Assign) and "SystemTrust.install" in ast.unparse(node.value):
                break
            if isinstance(node, ast.Import):
                imported_before |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported_before.add((node.module or "").split(".")[0])
        else:
            pytest.fail("floxy_ai_server.py never calls SystemTrust.install() at module level")

        assert imported_before <= harmless, f"imported before the trust store is installed: {imported_before - harmless}"
