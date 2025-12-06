import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import requests
import sys
import os
import signal

# Define a fixture to start the panel server
@pytest.fixture(scope="module")
def panel_server():
    # Start the panel app
    cmd = [sys.executable, "-m", "panel", "serve", "dashboard/app.py", "--port", "5006", "--allow-websocket-origin", "*"]
    # We use setsid to be able to kill the whole process group later
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    url = "http://localhost:5006/app"

    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        # Failed to start
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        pytest.fail("Panel server failed to start")

    yield url

    # Cleanup
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def test_dashboard_loads(page: Page, panel_server):
    page.goto(panel_server)

    # Check title
    expect(page).to_have_title("MD Simulation & Analysis Dashboard")

    # Check if "Configuration" tab is visible
    # Panel uses shadow DOM or specific structures, but text should be visible
    expect(page.get_by_text("MD Simulation Configuration")).to_be_visible()

    # Check widgets
    expect(page.get_by_role("button", name="Run Simulation")).to_be_visible()

    # Switch to Visualization tab
    # Tabs are usually buttons or links with text "Visualization"
    page.get_by_text("Visualization").click()

    # Check if plots are loading (look for canvas or plotly class)
    # This might be tricky as Plotly renders in iframes or complex DOM.
    # We check for text that is in the layout
    expect(page.get_by_text("Limit Extracted Structures")).to_be_visible()

def test_run_simulation(page: Page, panel_server):
    page.goto(panel_server)

    # Change temperature
    # Finding the input might be tricky with standard selectors,
    # Panel widgets have specific structures.
    # We can try to locate by label "Temperature (K)"

    # Run simulation
    page.get_by_role("button", name="Run Simulation").click()

    # Check status text
    # It should change to "Running..." immediately
    expect(page.get_by_text("Running...")).to_be_visible()

    # Then "Simulation Complete!"
    # We might need to wait (Mock takes 5s + overhead)
    expect(page.get_by_text("Simulation Complete!")).to_be_visible(timeout=20000)
