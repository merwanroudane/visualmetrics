"""Shared fixtures.

Tests are grouped by what they protect:

``unit``          the core dataclasses and the numeric backend;
``integration``   the catalog, the registry and the labs end to end;
``scientific``    numerical identities and statistical properties;
``translations``  trilingual bundle alignment and RTL handling;
``regression``    the honesty rules that must never quietly lapse.
"""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG


@pytest.fixture(scope="session")
def catalog():
    """Every catalogued concept stub, implemented or planned."""
    return CATALOG


@pytest.fixture(scope="session")
def implemented_ids() -> list[str]:
    return [s.id for s in vm.registry.specs(implemented_only=True)]


@pytest.fixture(scope="session")
def planned_ids() -> list[str]:
    return [s.id for s in CATALOG if not s.status.is_implemented]


@pytest.fixture(autouse=True)
def _reset_language():
    """Never let one test's language leak into the next."""
    from visualmetrics.i18n.translator import set_language

    set_language("en")
    yield
    set_language("en")


def pytest_collection_modifyitems(config, items):
    """Tests placed under ``tests/slow`` or naming a full sweep are marked slow."""
    for item in items:
        if "sweep" in item.name or "every_scenario" in item.name:
            item.add_marker(pytest.mark.slow)
