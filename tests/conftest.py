import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-published-models",
        action="store_true",
        default=False,
        help="run opt-in tests that download published Hugging Face checkpoints",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-published-models"):
        return

    skip = pytest.mark.skip(reason="use --run-published-models to download checkpoints")
    for item in items:
        if "published_models" in item.keywords:
            item.add_marker(skip)
