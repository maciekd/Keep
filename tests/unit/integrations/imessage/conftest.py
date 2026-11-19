"""The iMessage integration's tests run at the moment their fakes were stamped."""
import time
from unittest.mock import Mock

import pytest

from . import host_fakes


@pytest.fixture(autouse=True)
def the_time_of_the_fakes(monkeypatch):
    """`host_fakes.RECENT` is stamped once, when a worker imports the module, and a 2FA code is
    taken only from a message under `TFA_CODE_MAX_AGE` old: on a run of two minutes the worker
    reached those tests too late. The integration reads `host_fakes.NOW` as the time instead."""
    from integrations.imessage import IMessageIntegration
    monkeypatch.setattr(IMessageIntegration, "time", Mock(wraps=time, **{"time.return_value": host_fakes.NOW}))
