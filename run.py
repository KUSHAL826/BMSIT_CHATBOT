import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.config import Config

app = create_app()

if __name__ == "__main__":
    host = Config.HOST
    port = Config.PORT
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host

    print("=" * 62)
    print(" BMSIT College AI Agent")
    print("=" * 62)
    print(f"  Chatbot   ->  http://{shown_host}:{port}/chatbot")
    print(f"  Admin     ->  http://{shown_host}:{port}/admin")
    print(f"  Health    ->  http://{shown_host}:{port}/health")
    print(f"  Bound to  ->  {host}:{port}")
    print(f"  Daily delta crawl at {Config.SCHEDULE_TIME} ({Config.SCHEDULE_TIMEZONE})")
    if host == "0.0.0.0":
        print("  NOTE: bound to all interfaces and /admin has no authentication.")
    print("=" * 62)

    app.run(host=host, port=port, debug=False, use_reloader=False)
