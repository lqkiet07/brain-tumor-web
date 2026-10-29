import os
import threading
import time
import webbrowser

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8010"))
URL = f"http://{HOST}:{PORT}"


def _open_browser() -> None:
    time.sleep(3.5)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    from server.app import app

    if not os.environ.get("NO_BROWSER"):
        threading.Thread(target=_open_browser, daemon=True).start()
    print(f"\n  Demo dang chay tai: {URL}\n  Nhan Ctrl+C de dung.\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
