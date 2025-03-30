# --------------------------------------
import numpy as np

# --------------------------------------
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.color import gray2rgba

# --------------------------------------
from bokeh.layouts import layout
from bokeh.models import Div
from bokeh.models import Row
from bokeh.models import Column
from bokeh.plotting import show
from bokeh.plotting import figure
from bokeh.plotting import output_file
from bokeh.plotting import output_notebook
from bokeh.plotting import curdoc

# --------------------------------------
from skimage.exposure import rescale_intensity

# --------------------------------------
import pyrception as pcp
import pyrception.utils.functions as pcf
from pyrception.utils.functions import is_notebook

if is_notebook():
    pcp.logger.info("Running inside a notebook.")
    output_notebook()
curdoc().theme = "dark_minimal"


class Plotter:

    def _figure(
        self,
        width: int = 640,
        height: int = 480,
        title: str = None,
        tools: str | set[str] | bool = True,
        logo: bool = False,
        title_style: dict | None = None,
    ) -> figure:
        """
        Create a Bokeh figure.

        Args:

            width (int, optional):
                Figure width. Defaults to 640.

            height (int, optional):
                Figure height. Defaults to 640.

            title (str, optional):
                The title for this figure. Defaults to None.

            tools (str | set[str] | bool, optional):
                Show one or more specific tools or hide the toolbox altogether. Defaults to True.

            logo (bool, optional):
                Show or hide the logo. Defaults to False.

            title_style (dict | None, optional):
                Title style. Defaults to None.

        Returns:
            figure:
                Returns a Bokeh figure.
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
            width=int(width),
            height=int(height),
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
        self,
        image: np.ndarray,
        title: str = None,
        scale: float = 1.0,
        greyscale: str = False,
        tools: str | set[str] | bool = True,
        logo: bool = False,
        title_style: dict | None = None,
    ) -> figure:
        """
        Plot an image.

        Args:
            image (np.ndarray):
                Image to display.

            title (str, optional):
                The title for this image. Defaults to None.

            greyscale (str, optional):
                Display the image in greyscale. Defaults to False.

            tools (str | set[str] | bool, optional):
                Show one or more specific tools or hide the toolbox altogether. Defaults to True.

            logo (bool, optional):
                Show or hide the logo. Defaults to False.

            title_style (dict | None, optional):
                Title style. Defaults to None.

        Returns:
            figure:
                Returns a Bokeh figure.
        """

        # Prepare the image
        # ==================================================
        if len(image.shape) == 3 and greyscale:
            image = rgb2gray(image)

        if len(image.shape) == 2:
            image = gray2rgb(image)

        image = rescale_intensity(image, out_range=np.uint8)
        if len(image.shape) == 3:
            image = np.dstack((image, np.full(image.shape[:2], 255, dtype=image.dtype)))

        (h, w) = image.shape[:2]

        e = np.empty((h, w), dtype=np.uint32)
        view = e.view(dtype=np.uint8).reshape((h, w, 4))

        view[:] = image

        # Prepare the figure
        # ==================================================
        p = self._figure(
            w * scale + 30,
            h * scale + 30,
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
        p.image_rgba(image=[e], x=0, y=0, dw=10, dh=10, origin="top_left")

        return p

    def scatter(
        self,
        ys: np.ndarray,
        xs: np.ndarray = None,
        width: int = 640,
        height: int = 480,
        title: str = None,
        xtitle: str = None,
        ytitle: str = None,
        tools: str | set[str] | bool = True,
        logo: bool = False,
        title_style: dict | None = None,
    ) -> figure:
        """
        Create a scatter plot.

        Args:
            ys (np.ndarray):
                Data to plot (Y axis).

            xs (np.ndarray, optional):
                Data to plot (X axis).
                If unset, defaults to np.arange(len(ys)).
                Defaults to None.

            width (int, optional):
                Figure width. Defaults to 640.

            height (int, optional):
                Figure height. Defaults to 640.

            title (str, optional):
                The title for this image. Defaults to None.

            xtitle (str, optional):
                The title for the x axis. Defaults to None.

            ytitle (str, optional):
                The title for the y axis. Defaults to None.

            tools (str | set[str] | bool, optional):
                Show one or more specific tools or hide the toolbox altogether. Defaults to True.

            logo (bool, optional):
                Show or hide the logo. Defaults to False.

            title_style (dict | None, optional):
                Title style. Defaults to None.

        Returns:
            figure:
                Returns a Bokeh figure.
        """

        # Prepare the figure
        # ==================================================
        p = self._figure(
            width,
            height,
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

        return p

    def show(
        self,
        entries: figure | list[figure],
        title: str | None = None,
        title_style: dict | None = None,
    ):
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
