"""Short original-source vs migrated FedFewS acceptance entry."""
import os
import sys

if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    if "--device-id" in sys.argv:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device-id") + 1]
    from fedfews_compat.acceptance import main
    main()
