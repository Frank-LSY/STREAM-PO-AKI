"""Backward-compatible entry. Prefer ``event_stream.transformer.train``."""
from event_stream.transformer.train import main, parse_args

if __name__ == "__main__":
    main()
