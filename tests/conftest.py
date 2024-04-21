import pytest

@pytest.fixture(scope="session")
def spi_warning_logger(request):
    warnings_log = list()

    def add_warning(spi, module_name, max_z, num_exceed, num_iteractions):
        warnings_log.append((spi, module_name, max_z, num_exceed, num_iteractions))
    
    request.session.spi_warnings = warnings_log
    return add_warning

def pytest_sessionfinish(session, exitstatus):
    # retrieve the spi warnings from the session object
    spi_warnings = getattr(session, 'spi_warnings', [])

    # styling
    header_line = "=" * 80
    content_line = "-" * 80
    footer_line = "=" * 80
    header = " SPI BENCHMARKING SUMMARY"
    footer = f" Session completed with exit status: {exitstatus} "
    padded_header = f"{header:^80}"
    padded_footer = f"{footer:^80}"

    print("\n")
    print(header_line)
    print(padded_header)
    print(header_line)

    # print problematic SPIs in table format
    if spi_warnings:
        print(f"\nDetected {len(spi_warnings)} SPI(s) with outputs exceeding the specified 1 sigma threshold.\n")

        # table header
        print(f"{'SPI':<25}{'Cat':<10}{'Max ZSc.':>10}{'# Exceed. Pairs':>20}{'Unq. Pairs':>15}")
        print(content_line)

        # table content
        for est, module_name, max_z, num_exceed, num_iteractions in spi_warnings:
            # add special character for v.large zscores
            error = ""
            if max_z > 10:
                error = " **"
            print(f"{est+error:<25}{module_name:<10}{max_z:>10.4g}{num_exceed:>15}{num_iteractions:>20}")
    else:
        print("\n\nNo SPIs exceeded the 1 sigma threshold.\n")

    print(footer_line)
    print(padded_footer)
