from nnp_gen.web_ui.app import build_app
import panel as pn

if __name__ == "__main__":
    app = build_app()
    pn.serve(app, port=5006, show=False)
