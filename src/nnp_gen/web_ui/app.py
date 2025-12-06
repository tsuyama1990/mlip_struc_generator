import panel as pn
from nnp_gen.web_ui.tabs.config_tab import ConfigTab
from nnp_gen.web_ui.tabs.viz_tab import VizTab

def build_app():
    pn.extension('bokeh', template='material')

    config_tab = ConfigTab()
    viz_tab = VizTab()

    tabs = pn.Tabs(
        ("Configuration", config_tab.view()),
        ("Visualization", viz_tab.view())
    )

    title = "MLIP Structure Generator"

    template = pn.template.MaterialTemplate(
        title=title,
        main=[tabs],
    )

    return template
