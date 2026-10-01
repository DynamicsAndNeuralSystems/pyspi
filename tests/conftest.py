import pytest


@pytest.fixture(scope="session")
def spi_warning_logger(request):
    """Collect (dataset, SPI) drift records for the session-end summary table."""
    warnings_log = list()

    def add_warning(spi, module_name, max_abs, max_rel, num_exceed, num_interactions):
        warnings_log.append((spi, module_name, max_abs, max_rel, num_exceed, num_interactions))

    request.session.spi_warnings = warnings_log
    return add_warning


def pytest_sessionfinish(session, exitstatus):
    # Only print when the drift suite actually ran and produced records. The
    # fixture is session-scoped and lazily instantiated, so `spi_warnings` is
    # absent for --collect-only and for the default (non-slow) suite, and empty
    # when the drift suite ran clean — in both cases the banner is noise.
    spi_warnings = getattr(session, "spi_warnings", None)
    if not spi_warnings:
        return

    header_line = "=" * 90
    content_line = "-" * 90
    footer_line = "=" * 90
    header = " SPI DRIFT SUMMARY (abs/rel tolerance vs frozen baseline) "
    footer = f" Session completed with exit status: {exitstatus} "

    print("\n")
    print(header_line)
    print(f"{header:^90}")
    print(header_line)

    print(f"\nDetected {len(spi_warnings)} (dataset, SPI) pair(s) exceeding their "
          f"family's drift tolerance.\n")
    print(f"{'Dataset:SPI':<40}{'Cat':<10}{'Max |Δ|':>12}{'Max rel':>12}"
          f"{'# Exceed':>10}{'Unq Pairs':>12}")
    print(content_line)

    for est, module_name, max_abs, max_rel, num_exceed, num_interactions in spi_warnings:
        marker = " **" if max_rel > 0.1 or max_abs > 0.1 else ""
        rel_str = f"{max_rel:>12.4g}" if max_rel == max_rel else f"{'n/a':>12}"
        print(f"{est + marker:<40}{module_name:<10}{max_abs:>12.4g}{rel_str}"
              f"{num_exceed:>10}{num_interactions:>12}")

    print(footer_line)
    print(f"{footer:^90}")
