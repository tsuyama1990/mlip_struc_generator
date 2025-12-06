import json

def generate_3dmol_html(xyz_data: str, width="100%", height="400px") -> str:
    """
    Generate HTML for embedding 3Dmol.js viewer.
    """
    # Use json.dumps to safely serialize the string for JavaScript injection.
    # This handles escaping of quotes, backslashes, and newlines correctly, prevents XSS.
    xyz_json = json.dumps(xyz_data)

    html = f"""
    <div style="height: {height}; width: {width}; position: relative;" class="viewer_container">
        <div id="3dmol-viewer" style="width: 100%; height: 100%; position: relative;"></div>
    </div>
    <script>
        (function() {{
            let viewer = $3Dmol.createViewer("3dmol-viewer", {{backgroundColor: "white"}});
            let xyz = {xyz_json};
            viewer.addModel(xyz, "xyz");
            viewer.setStyle({{stick: {{}}, sphere: {{scale: 0.3}}}});
            viewer.zoomTo();
            viewer.render();
        }})();
    </script>
    """
    return html

def get_3dmol_header() -> str:
    """Return script tag for CDN."""
    return '<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>'
