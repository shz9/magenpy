import logging

from magenpy.utils.bin_utils import setup_cli_logger


def _remove_handlers(*loggers):
    for logger in loggers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_cli_logger_hides_info_at_warning_level(capsys):
    package_logger = logging.getLogger("magenpy_test.cli_logging.package")
    cli_logger = logging.getLogger("magenpy_test_cli")

    try:
        setup_cli_logger(
            cli_logger, "WARNING", package_modules=["magenpy_test.cli_logging"]
        )

        package_logger.info("hidden package detail")
        package_logger.warning("visible package warning")
        cli_logger.info("hidden CLI status")
        cli_logger.warning("visible CLI warning")

        stderr = capsys.readouterr().err
        assert "hidden package detail" not in stderr
        assert "hidden CLI status" not in stderr
        assert "visible package warning" in stderr
        assert "visible CLI warning" in stderr
    finally:
        _remove_handlers(package_logger, cli_logger)


def test_cli_logger_can_enable_package_info(capsys):
    package_parent = logging.getLogger("magenpy_test.verbose_logging")
    package_logger = logging.getLogger("magenpy_test.verbose_logging.package")
    cli_logger = logging.getLogger("magenpy_test_verbose_cli")

    try:
        setup_cli_logger(
            cli_logger, "INFO", package_modules=["magenpy_test.verbose_logging"]
        )
        package_logger.info("requested package detail")
        cli_logger.info("requested CLI detail")

        stderr = capsys.readouterr().err
        assert stderr.count("requested package detail") == 1
        assert stderr.count("requested CLI detail") == 1
    finally:
        _remove_handlers(package_parent, package_logger, cli_logger)
