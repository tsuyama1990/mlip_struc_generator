import json

def generate_3dmol_html(xyz_data: str, width="100%", height="400px") -> str:
    """
    Generate HTML for embedding 3Dmol.js viewer.
    """
    # Use json.dumps to safely serialize the string for JavaScript injection.
    # This handles escaping of quotes, backslashes, and newlines correctly, prevents XSS.
    import uuid
    # Use a simpler ID format to avoid potential selector issues, but keep uniqueness
    raw_uid = uuid.uuid4().hex
    unique_id = f"viewer_{raw_uid}"
    xyz_json = json.dumps(xyz_data)

    # Use unique_id as BOTH id and class to ensure findability
    html = f"""
    <div style="height: {height}; width: {width}; position: relative;" class="viewer_container">
        <div id="{unique_id}" class="{unique_id}" style="width: 100%; height: 100%; position: relative;"></div>
    </div>
    <script>
        (function() {{
            var xyz_data = {xyz_json};
            var unique_id = "{unique_id}";
            
            function initViewer(element) {{
                try {{
                    let viewer = $3Dmol.createViewer(element, {{backgroundColor: "white"}});
                    viewer.addModel(xyz_data, "xyz");
                    viewer.setStyle({{stick: {{}}, sphere: {{scale: 0.3}}}});
                    viewer.zoomTo();
                    viewer.render();
                }} catch (e) {{
                    console.error("3Dmol init failed for " + unique_id + ": " + e);
                }}
            }}

            // Robust element finding strategy
            function findElement(id) {{
                // 1. Try standard ID lookup
                var el = document.getElementById(id);
                if (el) return el;

                // 2. Try Class Name lookup (Robust against ID stripping)
                var byClass = document.getElementsByClassName(id);
                if (byClass && byClass.length > 0) return byClass[0];

                // 3. Shadow DOM piercing (Recursive)
                function searchShadow(root) {{
                    var found = root.getElementById(id);
                    if (found) return found;
                    
                    var foundClass = root.querySelector('.' + id);
                    if (foundClass) return foundClass;

                    var all = root.querySelectorAll('*');
                    for (var i = 0; i < all.length; i++) {{
                        if (all[i].shadowRoot) {{
                            var res = searchShadow(all[i].shadowRoot);
                            if (res) return res;
                        }}
                    }}
                    return null;
                }}
                
                // Start shadow search from document body
                return searchShadow(document);
            }}

            var checkCount = 0;
            var maxChecks = 50; 
            var interval = setInterval(function() {{
                var element = findElement(unique_id);
                if (element) {{
                    clearInterval(interval);
                    initViewer(element);
                }} else {{
                    checkCount++;
                    if (checkCount >= maxChecks) {{
                        clearInterval(interval);
                        console.error("3Dmol viewer element not found: " + unique_id);
                    }}
                }}
            }}, 50);
        }})();
    </script>
    """
    return html

def get_3dmol_header() -> str:
    """Return script tag for CDN."""
    return '<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>'
