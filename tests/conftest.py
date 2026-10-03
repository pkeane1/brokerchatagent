import pytest

from brokerchat.glossary import Glossary


@pytest.fixture(scope="session")
def glossary() -> Glossary:
    return Glossary.load()
