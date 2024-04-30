from typing import *

# --------------------------------------
import cv2 as cv

# --------------------------------------
import numpy as np

# --------------------------------------
from pathlib import Path

# --------------------------------------
import h5py

# --------------------------------------
import shutil

# --------------------------------------
from tqdm import tqdm

# --------------------------------------
from scipy.sparse import coo_matrix

# --------------------------------------
import subprocess


class EventLoader:
    """
    HDF5 data class.

    This class should contain methods to load, convert, save and display image data from HDF5 files.
    H5Data objects can be included in event processing pipelines.
    """

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the (NumPy) data contained in the HDF5 file.

        Returns:
            Tuple[int]:
                The shape of the data as a tuple of integers.
        """
        return self._data.shape if self._data is not None else None

    @staticmethod
    def to_path(
        path: Union[str, Path],
        create: bool = False,
    ):
        path = Path(path).absolute()

        if not path.is_file():
            raise SystemExit(f"The specified file '{path}' is not a file. Exiting.")

        if not path.exists():
            if not create:
                raise SystemExit(
                    f"The specified file '{path}' does not exist. Exiting."
                )

        return path

    @staticmethod
    def to_dir(
        root: Union[str, Path],
        subdir: Union[str, Path] = None,
        clean: bool = False,
    ):
        if subdir is None:
            return

        subdir = (Path(root) / subdir).absolute()

        if subdir.is_file():
            raise SystemExit(f"The specified path '{subdir}' is a file. Exiting.")

        if subdir.exists():
            if clean:
                shutil.rmtree(subdir)

        subdir.mkdir(parents=True, exist_ok=True)

        return subdir

    @staticmethod
    def load_h5(path: Union[str, Path]):
        path = EventLoader.to_path(path)

        f = h5py.File(path, "r")

        events = f["CD"]["events"]
        triggers = f["EXT_TRIGGER"]["events"] if "EXT_TRIGGER" in f.keys() else None

        return EventLoader(events, triggers, path.parent / path.stem)

    @staticmethod
    def load_raw(
        path: Union[str, Path],
        save: bool = True,
    ):
        path = EventLoader.to_path(path)
        h5path = path.with_suffix(".hdf5")

        if not h5path.exists():
            if not save:
                raise SystemExit(f"File '{h5path}' does not exist, exiting.")

            print(f"==[ Converting RAW to HDF5...")
            subprocess.call(["metavision_file_to_hdf5", "-i", f"{path}"])

        return EventLoader.load_h5(h5path)

    def __init__(
        self,
        events: np.ndarray,
        triggers: np.ndarray = None,
        root: Union[str, Path] = None,
    ):
        """
        H5 converter.

        Args:
            path (Union[str, Path]):
                Path to the HDF5 file.

            keys (Optional[Union[str, List[str]]], optional):
                A key or a list of keys specifying the data.
                Defaults to None.

        Raises:
            ValueError:
                Invalid path provided.
        """

        self.events = events
        self.triggers = triggers
        self.root = root

        # TODO: Extract those from the metadata
        # ==================================================
        self.height = 720
        self.width = 1280

    def segment(
        self,
        duration: int,  # us
        offset: int = 0,  # us
        limit: int = None,
        use_triggers: bool = True,
        save: str = False,
        clean: bool = False,
    ):
        # Sanity checks and initialisations
        # ==================================================
        if use_triggers:
            if self.triggers is None:
                raise SystemExit(
                    f"Triggers requested but the data does not contain triggers. Exiting."
                )

            first_ts = self.triggers[0][1]

        else:
            first_ts = self.events[0][-1]

        event_dir = None
        image_dir = None
        if save:
            event_dir = EventLoader.to_dir(
                self.root,
                f"segments/offset_{offset}/duration_{duration}/events",
                clean=clean,
            )
            image_dir = EventLoader.to_dir(
                self.root,
                f"segments/offset_{offset}/duration_{duration}/images",
                clean=clean,
            )

        # Seek the first and last event
        # ==================================================
        events = self.events
        first_ts += offset
        events = events[self.events["t"] >= first_ts]
        last_ts = events[-1][-1]

        if limit is None:
            limit = np.inf

        # Extract the segments
        # ==================================================
        segments = []
        count = 0
        while first_ts <= last_ts and count < limit:
            segment = events[
                np.logical_and(
                    first_ts <= events["t"], events["t"] < first_ts + duration
                )
            ]
            segments.append(segment)
            first_ts += duration
            count += 1

        if event_dir is not None:
            print(f"Saving .npz archives to {event_dir}...")
            for idx, segment in enumerate(tqdm(segments), 1):
                np.savez_compressed(event_dir / f"{idx:06d}.npz", segment)

        if image_dir is not None:
            print(f"Saving images to {image_dir}...")
            for idx, segment in enumerate(tqdm(segments), 1):
                buffer = np.array([[*ev] for ev in segment], dtype=np.int32)
                (x, y, p) = buffer[:, 0], buffer[:, 1], buffer[:, 2]
                p[p == 0] = -1

                buffer = (
                    coo_matrix(
                        (p, (x, y)),
                        shape=(
                            self.width,
                            self.height,
                        ),
                    )
                    .toarray()
                    .T
                )

                buffer = (2**16 - 1) * (
                    (buffer - buffer.min()) / (buffer.max() - buffer.min())
                )

                cv.imwrite(str(image_dir / f"{idx:06d}.png"), buffer.astype(np.uint16))
