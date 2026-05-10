"""Automated screenshot capture of the running Streamlit app."""
import subprocess
import time
import sys
import signal
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url: str, timeout: int = 60) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    port = 8501
    url = f"http://localhost:{port}"

    # Kill any existing Streamlit on this port
    os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")
    time.sleep(1)

    # Start Streamlit in background
    print("Starting Streamlit...")
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"
    env["SHOWCASE_MODE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py"), "--server.port", str(port), "--server.headless", "true"],
        stdout=open("/tmp/streamlit_screenshot.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        if not wait_for_server(url, timeout=60):
            print("Streamlit did not start in time.")
            sys.exit(1)

        print("Streamlit ready. Capturing screenshots...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            # Load page and wait for engine init
            page.goto(url)
            page.wait_for_load_state("networkidle")
            # Wait for the chat history to appear (showcase mode pre-populates it)
            page.wait_for_selector("[data-testid='stChatMessage']", timeout=120000)
            time.sleep(3)

            # Screenshot 1: landing with chat visible
            page.screenshot(path=str(SCREENSHOTS_DIR / "01_landing.png"))
            print("Captured 01_landing.png")

            # Click the Sources expander in the first assistant message
            page.locator("summary:has-text('Sources')").first.click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_chat_with_sources.png"))
            print("Captured 02_chat_with_sources.png")

            # Scroll sidebar to show settings and take another screenshot
            page.locator("[data-testid='stSidebar']").evaluate("el => el.scrollTo(0, el.scrollHeight)")
            page.wait_for_timeout(800)
            page.screenshot(path=str(SCREENSHOTS_DIR / "03_sidebar_settings.png"))
            print("Captured 03_sidebar_settings.png")

            browser.close()
    finally:
        print("Shutting down Streamlit...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        print(f"Screenshots saved to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
