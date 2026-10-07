"""SystemTrust.install(), the unverified context that stays usable after it, and the launcher's promise to
call install() before anything can build an SSL context."""
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
class TestUnverifiedContext:
    def test_it_verifies_nothing_and_is_not_truststores(self, stock_ssl):
        import truststore
        SystemTrust.install()

        context = SystemTrust.unverified_context()

        assert not isinstance(context, truststore.SSLContext)
        assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False

    def test_its_setters_work_while_the_swap_is_in_place(self, stock_ssl):
        # urllib3 sets verify_mode on the context it is handed; the stock class's setter recurses
        # once ssl.SSLContext names truststore's class.
        SystemTrust.install()
        context = SystemTrust.unverified_context()

        context.verify_mode = ssl.CERT_REQUIRED
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.options |= ssl.OP_NO_COMPRESSION

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
        assert context.options & ssl.OP_NO_COMPRESSION

    def test_the_same_context_comes_out_before_the_swap(self, stock_ssl):
        context = SystemTrust.unverified_context()
        context.verify_mode = ssl.CERT_OPTIONAL
        assert context.verify_mode == ssl.CERT_OPTIONAL
        assert isinstance(context, ssl.SSLContext)


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
