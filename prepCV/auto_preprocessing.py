from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Optional, Type

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .utils import get_cv2_function_params, parameter_combinations


class Preprocessor:
    """
    Is constructed from the resolved PipelineDescription and executes all preprocessing steps
    in the same order as inside the dictionary.

    Resolved PipelineDescription contain only SINGLE value for each parameter used in a function:
    description = {cv2.adaptiveThreshold: {'maxValue': 255,
                                           'adaptiveMethod' : cv2.ADAPTIVE_THRESH_GAUSSIAN},
               cv2.dilate : {'kernel': np.ones((3,3))}
              }
    """

    def __init__(self, description: PipelineDescription):
        self.description = description

    def process(self, np_image: np.ndarray) -> np.ndarray:
        image = np_image.copy()
        for function, params in self.description.description.items():
            image = function(image, **params)
        return image

    def __str__(self) -> str:
        return f"Preprocessor containing {self.description.__repr__()}"


@dataclass(frozen=True)
class PipelineDescription:
    """
    Interface for describing pipelines and defining for them.
    Automatically validates incoming pipelines, making sure that every specified function can be
    called with all set of listed parameters.

    Accepts dictionaries in init, formatted like following:
    description = {cv2.adaptiveThreshold: {'maxValue': [255],
                                           'adaptiveMethod' : [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C]},
                   cv2.dilate : {'kernel': [...],
                                 'dst' : [...],}
                  }
    """

    description: dict[Callable[..., Any], dict[str, list[Any] | Any]]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for function, param_dict in self.description.items():
            valid_params = get_cv2_function_params(function)
            if not valid_params:
                # Optionally, warn the user and skip validation for this function
                print(
                    f"Warning: Unable to retrieve parameters for function '{function.__name__}'. Skipping validation."
                )
                continue

            # Check for invalid parameter names
            for param_name in param_dict.keys():
                if param_name not in valid_params:
                    raise ValueError(
                        f"Invalid parameter '{param_name}' for function '{function.__name__}'. "
                        f"Valid parameters are: {', '.join(valid_params)}"
                    )


class OcrEngine(ABC):
    @abstractmethod
    def process(self, np_image: np.ndarray) -> np.ndarray:
        pass


class PipelineManager:
    """
    A singletone object which stores all the user-defined pipelines,
    launches competition between them using parameter GridSearch and visualuzation (via Search class),
    cache the competition results and constructs the actual working preprocessors from their descriptions.
    """

    pipelines: list[Preprocessor] = []
    best_preprocessor = None
    newly_added: list[Preprocessor] = []

    @classmethod
    def _resolve_complex_descriptions(
        cls, pipeline_description: PipelineDescription
    ) -> list[PipelineDescription]:
        """
        Generates PipelineDescriptions for each possible parameter combination.
        """
        resolved_pipelines = []
        function_param_combinations = []

        for function, parameter_dict in pipeline_description.description.items():
            param_combinations_for_function = [
                {function: combination} for combination in parameter_combinations(parameter_dict)
            ]
            function_param_combinations.append(param_combinations_for_function)

        for pipeline_combination in product(*function_param_combinations):
            resolved_pipeline = {}
            for function_params in pipeline_combination:
                resolved_pipeline.update(function_params)
            resolved_pipelines.append(PipelineDescription(resolved_pipeline))

        return resolved_pipelines

    @classmethod
    def add_pipeline(cls, pipeline_description: PipelineDescription):
        for resolved_description in cls._resolve_complex_descriptions(pipeline_description):
            cls.pipelines.append(Preprocessor(resolved_description))
            cls.newly_added.append(Preprocessor(resolved_description))

    @classmethod
    def run_search(
        cls,
        np_image: np.ndarray,
        search_strategy: str,
        ocr_engine: Optional[OcrEngine] = None,
        cold_start=False,
    ):
        valid_strategies = "GridSearch"
        if search_strategy == "GridSearch":
            strategy = GridSearch(cls, ocr_engine, cold_start)

        else:
            print(f"No such strategy implemented. Valid options are: {valid_strategies}")

        cls.best_preprocessor = strategy.search(np_image)
        cls.newly_added = []

    @classmethod
    def get_best_preprocessor(cls):
        if cls.newly_added:
            cls.best_preprocessor = None
            print("New pipelines were added. Do run_search once again to define a new winner")
        return cls.best_preprocessor


class SearchStrategy(ABC):
    def __init__(self, pipeline_manager: Type[PipelineManager], ocr_engine=None, cold_start=False):
        self.pipeline_manager = pipeline_manager
        self.ocr_engine = ocr_engine
        self.cold_start = cold_start

    @abstractmethod
    def search(self, np_image: np.ndarray) -> np.ndarray:
        pass


class GridSearch(SearchStrategy):
    """
    Best preprocessor search strategy must have a reference to a
    pipeline manager, and, optionally, to an OCR_engine used for bounding boxes detection
    """

    def search(self, np_image: np.ndarray) -> Preprocessor:
        competitors = (
            self.pipeline_manager.pipelines
            if self.cold_start
            else self.pipeline_manager.newly_added
        )
        competing_images = [preprocessor.process(np_image) for preprocessor in competitors]

        if self.ocr_engine:
            competing_images = [self.ocr_engine.process(np_image) for np_image in competing_images]

        best_image_index = ImageSelector.select_best_image(competing_images)
        assert best_image_index
        return competitors[best_image_index]


class ImageSelector:
    """
    Provides an interactive way to select the best image from a list (Singleton-like).

    This class uses class methods and variables to ensure that only one
    selection process is active at a time, avoiding unnecessary object instantiation.

    Example:
         best_image_index = ImageSelector.select_best_image(images, batch_size=4)
         print(f"Best image index: {best_image_index}")
    """

    _images: list[Any] = []
    _image_indexes: list[int] = []
    _current_batch_indexes: list[int] = []
    _best_image_index: int | None = None
    _batch_size: int = 4
    _fig: Optional[plt.figure] = None
    _axs: Optional[plt.axes] = None

    @classmethod
    def select_best_image(cls, images: list[Any], batch_size: int = 4) -> int | None:
        """
        Starts the interactive image selection process.

        Display images in batches and allows the user to select the best image
        from each batch until all images have been compared.

        Args:
            images (list[Any]): List of images to display.
            batch_size (int, optional): Number of images to display per batch. Defaults to 4.

        Returns:
            int: The index of the best image in the original 'images' list, or None if no selection is made.
        """
        cls._images = images
        cls._image_indexes = list(range(len(images)))
        cls._best_image_index = None
        cls._batch_size = batch_size

        # No competition required if there's only one image
        if len(cls._images) == 0:
            return None

        elif len(cls._images) == 1:
            cls._best_image_index = 0
            return cls._best_image_index

        while len(cls._image_indexes) > 0:
            cls._set_figure_and_axs()
            cls._show_next_batch()

        plt.close(cls._fig)
        return cls._best_image_index

    @classmethod
    def _show_next_batch(cls):
        """
        Displays the next batch of images and handles user input.
        """
        # Get next batch of indexes
        if cls._best_image_index is not None:
            batch_indexes = [cls._best_image_index] + cls._image_indexes[: cls._batch_size - 1]
            cls._image_indexes = cls._image_indexes[cls._batch_size - 1 :]
        else:
            batch_indexes = cls._image_indexes[: cls._batch_size]
            cls._image_indexes = cls._image_indexes[cls._batch_size :]

        # Check if figures are initialized correctly
        assert cls._fig is not None
        assert cls._axs is not None

        # Display images for the current batch
        for i, index in enumerate(batch_indexes):
            img = cls._images[index]

            # Convert from BGR to RGB before displaying
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            row, col = divmod(i, cls._axs.shape[0])

            cls._axs[row, col].set_title(f"Image {i + 1}")

            # Add black border around the image
            rect = patches.Rectangle(
                (0, 0),
                img.shape[1],
                img.shape[0],
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            cls._axs[row, col].add_patch(rect)

            cls._axs[row, col].axis("off")
            cls._axs[row, col].imshow(img)

        cls._current_batch_indexes = batch_indexes

        # Bind the key press event
        cls._fig.canvas.mpl_connect("key_press_event", cls._on_key)
        plt.show(block=True)  # Block to wait for user interaction

    @classmethod
    def _on_key(cls, event):
        """
        Handles keyboard input for image selection.
        """
        if event.key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            selected_index = int(event.key) - 1

            assert (
                cls._current_batch_indexes is not None
            ), "current_batch_indexes attribute is not initialized"
            cls._best_image_index = cls._current_batch_indexes[selected_index]
            plt.close()  # Close the plot after selection

        elif event.key.lower() == "c":
            plt.close()

    @classmethod
    def _set_figure_and_axs(cls):
        """
        Sets up the Matplotlib figure and axes for displaying images.
        """
        cls._fig, cls._axs = cls._create_subplots(batch_size=cls._batch_size)
        cls._fig.text(
            0.5,
            0.01,
            "Enter the corresponding [1-9] key or press 'C' to exit",
            ha="center",
            fontsize=12,
        )
        plt.subplots_adjust(bottom=0.2, wspace=0.1, hspace=0.1)
        plt.tight_layout()

    @staticmethod
    def _create_subplots(batch_size: int):
        """
        Creates a 2x2 or 3x3 subplot grid based on batch size.

        Args:
            batch_size: The number of plots to create.

        Returns:
            A tuple of the figure and axes objects.
        """

        # Matplotlib backend
        matplotlib.use("TkAgg")

        if batch_size <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 3

        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        return fig, axs


def main():
    def crop_image(image, minx, maxx, miny, maxy):
        """Crops an image using relative coordinates (0-1)."""
        height, width = image.shape[:2]
        x_start = int(width * minx)
        x_end = int(width * maxx)
        y_start = int(height * miny)
        y_end = int(height * maxy)
        return image[y_start:y_end, x_start:x_end]

    def resize_image(img, scale_factor):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        # resize image
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    pipeline_manage = PipelineManager()

    # Example Usage
    pipeline1 = PipelineDescription(
        {
            cv2.cvtColor: {"code": [cv2.COLOR_BGR2GRAY]},
            cv2.adaptiveThreshold: {
                "maxValue": [255],
                "adaptiveMethod": [cv2.ADAPTIVE_THRESH_MEAN_C],
                "thresholdType": [cv2.THRESH_BINARY],
                "blockSize": [5, 11, 13, 15],
                "C": [5],
            },
            crop_image: {"minx": [0.1], "maxx": [0.7], "miny": [0.4], "maxy": [0.95]},
            resize_image: {
                "scale_factor": [2, 3],
            },
            cv2.dilate: {"kernel": [np.ones((3, 3), int)]},
            cv2.erode: {"kernel": [np.ones((3, 3), int)]},
        }
    )

    pipeline2 = PipelineDescription(
        {
            cv2.cvtColor: {"code": [cv2.COLOR_BGR2GRAY]},
            cv2.adaptiveThreshold: {
                "maxValue": [255],
                "adaptiveMethod": [cv2.ADAPTIVE_THRESH_MEAN_C],
                "thresholdType": [cv2.THRESH_BINARY],
                "blockSize": [5, 11],
                "C": [10, 20, 30, 40],
            },
            crop_image: {"minx": [0.1], "maxx": [0.7], "miny": [0.4], "maxy": [0.95]},
        }
    )

    test_image = cv2.imread("./test_images/test_image1.png")
    pipeline_manage.add_pipeline(pipeline1)
    pipeline_manage.add_pipeline(pipeline2)
    pipeline_manage.run_search(test_image, "GridSearch")
    print(pipeline_manage.best_preprocessor)


if __name__ == "__main__":
    main()
