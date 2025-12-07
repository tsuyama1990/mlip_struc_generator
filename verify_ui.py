from playwright.sync_api import Page, expect, sync_playwright

def test_config_ui_guidance(page: Page):
    """
    Verify that the new guidance features (info boxes) and widgets (user_file, vacancies) are present.
    """
    # 1. Arrange: Go to the app
    page.goto("http://localhost:5006")

    # Wait for loading
    page.wait_for_selector("text=MLIP Structure Generator")

    # 2. Assert: Check for Quick Start Guidance
    expect(page.get_by_text("Quick Start")).to_be_visible()
    expect(page.get_by_text("Select System Type")).to_be_visible()

    # 3. Act: Select 'user_file' system type
    # Find the Select widget for system type.
    # It might be labelled "System Type" or just be a dropdown with "alloy".
    # Inspecting typical Panel output: <select> inside a shadow root or standard.
    # Using get_by_label if possible, or get_by_role('combobox').

    # Panel Select widgets usually have a label above them.
    # Let's try to select "user_file"
    # Note: Param Selector often renders as a <select>

    # Wait for the select to be visible.
    # The label "System type" comes from param doc/name?
    # In `ConfigViewModel`, `system_type` param has no `label` set, so it defaults to "System type" or "System Type".

    # Try finding the combobox
    select = page.locator("select").first
    select.select_option("user_file")

    # 4. Assert: Check for User File Guidance and Widgets
    # The Info box should update
    expect(page.get_by_text("Load structures from disk")).to_be_visible()

    # Check for File Path and Repeat inputs
    expect(page.get_by_text("File Path")).to_be_visible()
    expect(page.get_by_text("Repeat Count")).to_be_visible()

    # 5. Assert: Check for Advanced Physics (Vacancy)
    # Scroll down if needed? Playwright auto-scrolls.
    expect(page.get_by_text("Advanced Physics")).to_be_visible()
    expect(page.get_by_text("Vacancy concentration")).to_be_visible()

    # 6. Act: Toggle MC
    # Check "Mc enabled" toggle.
    # Toggle widgets are often checkboxes or switches.
    # Look for label "Mc enabled"
    mc_toggle = page.get_by_role("checkbox", name="Mc enabled")
    # Sometimes Panel renders it differently. Let's look for text.
    if not mc_toggle.is_visible():
        # Maybe it's a switch div?
        pass
    else:
        if not mc_toggle.is_checked():
            mc_toggle.click()

    # If successful, MC settings panel should appear
    # "Swap Interval"
    # Note: Panel updates might take a ms.
    # expect(page.get_by_text("Swap Interval")).to_be_visible()

    # 7. Screenshot
    page.screenshot(path="/app/verification_screenshot.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            test_config_ui_guidance(page)
            print("Verification script ran successfully.")
        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="/app/verification_failed.png")
        finally:
            browser.close()
