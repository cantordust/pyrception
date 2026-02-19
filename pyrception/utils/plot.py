
import numpy as np
from bokeh.models import Div
from bokeh.models import Row
from bokeh.layouts import layout
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from bokeh.plotting import show
from bokeh.plotting import curdoc
from bokeh.plotting import figure
from bokeh.plotting import output_notebook
from skimage.exposure import rescale_intensity

from pyrception import logger
from pyrception.utils.functions import is_notebook

from bokeh.io.notebook import CommsHandle

if is_notebook():
    logger.debug("Running inside a notebook.")
    output_notebook()
curdoc().theme = "dark_minimal"


def make_figure(
    height: int = 600,
    width: int = 800,
    title: str = None,
    tools: str | set[str] | bool = True,
    logo: bool = False,
    title_style: dict | None = None,
) -> figure:
    """
    Create a Bokeh figure.

    Args:

        height: Figure height.
        width: Figure width.
        title: The title for this figure.
        tools: Show one or more specific tools or hide the toolbox altogether.
        logo: Show or hide the logo.
        title_style: Title style.

    Returns:
        A Bokeh figure.
    """

    # Tools
    # ==================================================
    if isinstance(tools, bool) and tools:
        tools = ["wheel_zoom", "pan", "reset"]

    if isinstance(tools, (tuple, list)):
        tools = set(tools)

    if tools and not is_notebook():
        tools.add("save")

    if tools:
        tools = list(tools)
    else:
        tools = []

    # Figure
    # ==================================================
    p = figure(
        height=int(height),
        width=int(width),
        output_backend="webgl",
        tools=tools,
        title=title,
    )

    if title_style is None:
        title_style = {
            "text_font_size": "16px",
            "align": "center",
        }

    if p.title is not None:
        for k, v in title_style.items():
            setattr(p.title, k, v)

    if not logo:
        if len(tools) == 0:
            p.toolbar_location = None
        p.toolbar.logo = None

    return p


def image(
    image: np.ndarray,
    title: str = None,
    scale: float = 1.0,
    greyscale: str = False,
    tools: str | set[str] | bool = True,
    logo: bool = False,
    title_style: dict | None = None,
    display: bool = False,
) -> figure:
    """
    Plot an image.

    Args:
        image: Image to display.
        title: The title for this image.
        greyscale: Display the image in greyscale.
        tools: Show one or more specific tools or hide the toolbox altogether.
        logo: Show or hide the logo.
        title_style: Title style.
        display: Show the image instead of returning a Bokeh figure.

    Returns:
        A Bokeh figure.
    """

    # Prepare the image
    # ==================================================
    if len(image.shape) == 3 and greyscale:
        image = rgb2gray(image)

    if len(image.shape) == 2:
        image = gray2rgb(image)

    image = rescale_intensity(image, out_range=np.uint8)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = np.dstack((image, np.full(image.shape[:2], 255, dtype=image.dtype)))

    (h, w) = image.shape[:2]

    canvas = np.empty((h, w), dtype=np.uint32)
    view = canvas.view(dtype=np.uint8).reshape((h, w, 4))
    view[:] = image

    # Prepare the figure
    # ==================================================
    p = make_figure(
        h * scale + 30,
        w * scale + 30,
        title,
        tools,
        logo,
        title_style,
    )

    # Remove some visual elements that are not necessary
    # for an image, such as the grid and the border.
    p.x_range.range_padding = 0
    p.y_range.range_padding = 0
    p.toolbar.autohide = True
    p.axis.visible = False
    p.grid.visible = False
    p.min_border = 2
    p.margin = 1

    # Finally, plot the image.
    # This method expects a list of images
    # ==================================================
    p.image_rgba(image=[canvas], x=0, y=0, dw=10, dh=10, origin="top_left")

    if display:
        return show_composite(p)
    return p


def scatter(
    ys: np.ndarray,
    xs: np.ndarray = None,
    height: int = 600,
    width: int = 800,
    title: str = None,
    xtitle: str = None,
    ytitle: str = None,
    tools: str | set[str] | bool = True,
    logo: bool = False,
    title_style: dict | None = None,
    display: bool = False,
) -> figure:
    """
    Create a scatter plot.

    Args:
        ys: Data to plot (Y axis).
        xs: Data to plot (X axis).
        height: Figure height.
        width: Figure width.
        title: The title for this image.
        xtitle: The title for the x axis.
        ytitle: The title for the y axis.
        tools: Show one or more specific tools or hide the toolbox altogether.
        logo: Show or hide the logo.
        title_style: Title style.
        display: Show the image instead of returning a Bokeh figure.

    Returns:
        A Bokeh figure.
    """

    # Prepare the figure
    # ==================================================
    p = make_figure(
        height,
        width,
        title=title,
        tools=tools,
        logo=logo,
        title_style=title_style,
    )

    p.toolbar.autohide = True

    if xs is None:
        xs = np.arange(len(ys))

    p.scatter(
        xs,
        ys,
        color="#00ffff",
        size=0.1,
    )

    if xtitle is not None:
        p.xaxis.axis_label = xtitle

    if ytitle is not None:
        p.yaxis.axis_label = ytitle

    if display:
        return show_composite(p)
    return p


def show_composite(
    entries: figure | list[figure],
    title: str | None = None,
    title_style: dict | None = None,
) -> CommsHandle | None:
    """
    Display a previously plotted entry.

    Args:
        entries: Entries to display.
        title: Title to use for the whole plot.
        title_style: Title style passed to the bokeh Div element.
    """
    if title_style is None:
        title_style = {
            "font-size": "24px",
            "text-align": "center",
            "width": "100%",
        }
    if title is not None:
        title = Row(
            Div(
                text=title,
                styles=title_style,
            ),
            styles={
                "width": "100%",
            },
        )

        entries = [title, [entries]]

    if isinstance(entries, list):
        entries = layout(children=entries)

    show(entries)
