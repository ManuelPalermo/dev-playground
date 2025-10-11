from pathlib import Path

import pytest
import torch

PROJECT_ROOT_FOLDER = Path(__file__).resolve().parent.parent
PROJECT_RESOURCES_FOLDER = PROJECT_ROOT_FOLDER / "resources"
PROJECT_SCRIPTS_FOLDER = PROJECT_ROOT_FOLDER / "scripts"
PROJECT_TESTS_FOLDER = Path(__file__).resolve().parent


@pytest.fixture(scope="session", name="project_resources_folder")
def fixture_project_resources_folder() -> Path:
    """Fixture to provide the resources folder of the project."""
    return PROJECT_RESOURCES_FOLDER


@pytest.fixture(scope="session", name="project_root_folder")
def fixture_project_root_folder() -> Path:
    """Fixture to provide the root folder of the project."""
    return PROJECT_ROOT_FOLDER


@pytest.fixture(scope="session", name="project_scripts_folder")
def fixture_project_scripts_folder() -> Path:
    """Fixture to provide the scripts folder of the project."""
    return PROJECT_SCRIPTS_FOLDER


@pytest.fixture(scope="session", name="project_tests_folder")
def fixture_project_tests_folder() -> Path:
    """Fixture to provide the root folder of the tests."""
    return PROJECT_TESTS_FOLDER


@pytest.fixture(
    scope="session",
    name="device",
    params=(["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]),
)
def fixture_devices(request: pytest.FixtureRequest) -> str:
    """Fixture to provide available devices for testing."""
    return request.param
