def generate_3dmol_html(xyz_data: str, width="100%", height="400px") -> str:
    """
    Generate HTML for embedding 3Dmol.js viewer.
    """
    # Escaping backticks or other special chars in xyz_data if necessary
    # Usually XYZ data is safe lines of text.
    xyz_safe = xyz_data.replace("`", "\`")

    html = f"""
    <div style="height: {height}; width: {width}; position: relative;" class="viewer_container">
        <div id="3dmol-viewer" style="width: 100%; height: 100%; position: relative;"></div>
    </div>
    <script>
        (function() {{
            let viewer = $3Dmol.createViewer("3dmol-viewer", {{backgroundColor: "white"}});
            let xyz = `{xyz_safe}`;
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
