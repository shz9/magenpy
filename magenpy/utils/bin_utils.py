import logging

from .system_utils import setup_logger


logger = logging.getLogger(__name__)


def print_cli_logo(mgp_module, description, left_padding=10, output_logger=None):
    """
    Log the magenpy ASCII logo for a command-line script.

    :param mgp_module: The imported ``magenpy`` module.
    :param description: A short description to display below the logo.
    :param left_padding: Number of spaces to use for logo padding.
    :param output_logger: Logger that should emit the logo.
    """
    logo = mgp_module.make_ascii_logo(desc=description, left_padding=left_padding)
    (output_logger or logger).info("\n%s\n", logo)


def setup_cli_logger(logger, log_level, package_modules=None):
    """
    Configure logging for command-line scripts.

    Apply the requested level to package and CLI messages. Package logger
    propagation is disabled to prevent duplicate output from parent handlers.

    :param logger: The logger for the active CLI script.
    :param log_level: Requested logging level for the active CLI script.
    :param package_modules: Package logger module-name filters.
    """
    package_modules = package_modules or ["magenpy"]

    package_loggers = [
        lgr
        for lgr_name, lgr in logging.root.manager.loggerDict.items()
        if any(module in lgr_name for module in package_modules)
        and isinstance(lgr, logging.Logger)
    ]
    old_package_handlers = set()
    for package_logger in package_loggers:
        for handler in package_logger.handlers[:]:
            package_logger.removeHandler(handler)
            old_package_handlers.add(handler)
        package_logger.propagate = False
    for handler in old_package_handlers:
        handler.close()

    setup_logger(modules=package_modules, log_level=log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = False

    setup_logger(
        loggers=[logger],
        log_format="%(message)s",
        log_level=log_level,
    )


def format_rows(rows):
    """
    Format key-value pairs as aligned CLI summary rows.

    :param rows: Iterable of ``(label, value)`` pairs.
    :return: A string with aligned rows.
    """
    rows = [(label, value) for label, value in rows if value is not None]
    if len(rows) < 1:
        return "  None"

    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label:<{width}} : {value}" for label, value in rows)


def format_sections(sections):
    """
    Format titled sections for a CLI summary block.

    :param sections: Iterable of ``(section_title, rows)`` pairs.
    :return: A formatted multi-line string.
    """
    return "\n\n".join(f"{title}\n{format_rows(rows)}" for title, rows in sections)


def format_cli_block(title, body, width=80):
    """
    Wrap a CLI information block in clean separators.

    :param title: The block title.
    :param body: The block body.
    :param width: The separator width.
    :return: A formatted multi-line string.
    """
    sep = "=" * width
    return f"{sep}\n{title}\n{sep}\n{body}\n{sep}"
