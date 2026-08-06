from app.core.chat_loop import run_chat_loop
from app.core.config import load_settings
from app.core.logging import configure_logging


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    run_chat_loop(settings)


if __name__ == "__main__":
    main()

