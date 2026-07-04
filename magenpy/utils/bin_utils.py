import logging

from .system_utils import setup_logger


def print_cli_logo(mgp_module, description, left_padding=10):
    """
    Print the magenpy ASCII logo for a command-line script.

    :param mgp_module: The imported ``magenpy`` module.
    :param description: A short description to display below the logo.
    :param left_padding: Number of spaces to use for logo padding.
    """
    logo = mgp_module.make_ascii_logo(desc=description, left_padding=left_padding)
    print(f"\n{logo}\n", flush=True)


def setup_cli_logger(logger, log_level, package_modules=None):
    """
    Configure logging for command-line scripts.

    Package loggers emit warnings and errors by default, while the active CLI
    logger emits clean user-facing messages at INFO level unless DEBUG is requested.

    :param logger: The logger for the active CLI script.
    :param log_level: Requested logging level for the active CLI script.
    :param package_modules: Package logger module-name filters.
    """
    package_modules = package_modules or ["magenpy"]
    setup_logger(modules=package_modules, log_level="WARNING")

    logger.handlers.clear()
    logger.propagate = False

    setup_logger(
        loggers=[logger],
        log_format="%(message)s",
        log_level=["INFO", log_level][logging.getLevelName(log_level) < logging.INFO],
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
