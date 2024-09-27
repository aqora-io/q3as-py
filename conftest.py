def pytest_addoption(parser):
    parser.addoption(
        "--benchmark",
        action="store_true",
        dest="benchmark",
        default=False,
        help="enable benchmarks",
    )
