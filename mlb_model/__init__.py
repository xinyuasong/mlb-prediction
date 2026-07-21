"""MLB forecasting model - modular rebuild (see AUDIT.md and README.md)."""
import logging


class _ConsoleNoiseFilter(logging.Filter):
    """Keep routine data-layer fallback chatter (missing pitcher data,
    skipped weather, unposted lineups...) OUT of the terminal so the slate
    table renders cleanly. Everything still lands in the logfile - the
    console only shows these loggers at ERROR+. Progress/status messages
    from other loggers (e.g. backtest day counts) pass through normally."""
    QUIET = ("mlb_model.data", "mlb_model.features")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(self.QUIET):
            return record.levelno >= logging.ERROR
        return True


def setup_logging(verbose: bool = False, logfile: str | None = "mlb_model.log"):
    """Console: warnings+ (info with --verbose), minus data-layer noise.
    Logfile: full detail always, so every run leaves an auditable record of
    exactly what data it used and what it fell back on."""
    root = logging.getLogger("mlb_model")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    if not verbose:  # --verbose means "show me everything", filter off
        ch.addFilter(_ConsoleNoiseFilter())
    root.addHandler(ch)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root
