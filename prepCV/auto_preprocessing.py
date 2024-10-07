from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from types import BuiltinFunctionType, BuiltinMethodType
from typing import Any, Callable, Optional

import cv2
import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from prepCV.utils import get_cv2_function_params, parameter_combinations


class CacheManager:
    CACHE_FILE = "_prepCV_cache.pkl"

    @classmethod
    def load_preprocessors_from_cache(cls) -> Optional[list[Preprocessor]]:
        try:
            with open(cls.CACHE_FILE, "rb") as f:
                cache_dict = dill.load(f)
                preprocessors: list[Preprocessor] = cache_dict["preprocessors"]

            return preprocessors

        except FileNotFoundError:
            print(
                f"Loading Preprocessors from cachefile {cls.CACHE_FILE} has failed.",
                "Check for file existence and integrity",
            )
            return None

    @classmethod
    def load_best_preprocessor_from_cache(cls) -> Optional[Preprocessor]:
        try:
            with open(cls.CACHE_FILE, "rb") as f:
                cache_dict = dill.load(f)
                best_preprocessor: Preprocessor = cache_dict["best_preprocessor"]
                return best_preprocessor

        except FileNotFoundError:
            print(
                f"Loading Preprocessors from cachefile {cls.CACHE_FILE} has failed.",
                "Check for file existence and integrity",
            )
            return None

    @classmethod
    def save_preprocessors_to_cache(
        cls, preprocessors: list[Preprocessor], best_preprocessor: Optional[Preprocessor]
    ):
        if not all([preprocessors, best_preprocessor]):
            print(
                "You have passed NoneType object within listed preprocessors or best preprocessor",
                "Can not dump NoneType objects, please make sure that your data is valid",
                sep="\n",
            )

        cache_dict = {"preprocessors": preprocessors, "best_preprocessor": best_preprocessor}

        """Save the PipelineManager to the cache."""
        with open(cls.CACHE_FILE, "wb") as file:
            dill.dump(cache_dict, file)


class Preprocessor:
    """
    Constructed from the resolved PipelineDescription and executes all preprocessing steps
    in the same order as they go in the dictionary.

    Resolved PipelineDescription contain only one SINGLE value for each parameter used in a function:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Preprocessor):
            return False
        return self.description == other.description

    def __hash__(self) -> int:
        return self.description.__hash__()

    def __str__(self) -> str:
        return f"Preprocessor containing {self.description.__repr__()}"


class PreprocessorFactory:
    """
    Factory class responsible for creating Preprocessor objects
    from PipelineDescription objects.
    """

    @staticmethod
    def create_from_description(pipeline_description: PipelineDescription) -> set[Preprocessor]:
        """
        Generates a list of Preprocessor objects from a PipelineDescription.
        """
        resolved_pipelines = set()
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
            resolved_pipelines.add(Preprocessor(PipelineDescription(resolved_pipeline)))

        return resolved_pipelines


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

    def __eq__(self, other: object) -> bool:
        """Custom equality comparison for PipelineDescription."""
        if not isinstance(other, PipelineDescription):
            return False
        return self._get_hashable_representation() == other._get_hashable_representation()

    def __hash__(self) -> int:
        """Custom hash function for PipelineDescription."""
        return hash(self._get_hashable_representation())

    def _get_hashable_representation(self) -> tuple[tuple[str, str], ...]:
        """Returns a hashable tuple representation of the description."""
        hashable_description = []
        for func, params in self.description.items():
            if isinstance(func, (BuiltinFunctionType, BuiltinMethodType)):
                # Use qualified name for built-in functions
                func_representation = func.__qualname__
            else:
                # Use co_code directly for comparison
                func_representation = str(func.__code__.co_code)

            params_representations = str(params)
            hashable_description.append((func_representation, params_representations))

        return tuple(hashable_description)

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
    """
    OcrEngine responsible for Box Detection and Drawing. Can be involved in a competition process
    so user would prefer an image relying on Ocr Detection result.
    To achieve so, the OcrEngine should return a modified numpy image from the "process" method, containing drawn
    bounding boxes.
    """

    @abstractmethod
    def process(self, np_image: np.ndarray) -> np.ndarray:
        pass


class PipelineManager:
    """
    A singleton object which stores all user-defined pipelines,
    launches competition between them using parameter GridSearch,
    caches the competition results and constructs working preprocessors from their descriptions.
    """

    caching = False
    pipelines: list[Preprocessor] = []
    best_preprocessor: Optional[Preprocessor] = None
    newly_added: list[Preprocessor] = []

    @classmethod
    def load_from_cache(cls):
        """Load the PipelineManager from the cache."""

        cached_preprocessors = CacheManager.load_preprocessors_from_cache()
        if cached_preprocessors:
            cls.pipelines = cached_preprocessors
            print("Loaded seen preprocessors from cache.")

        best_preprocessor = CacheManager.load_best_preprocessor_from_cache()
        if best_preprocessor:
            cls.best_preprocessor = best_preprocessor
            print("Loaded best preprocessor from cache.")

    @classmethod
    def save_to_cache(cls):
        preprocessors_to_save = cls.pipelines
        if cls.newly_added:
            print(
                "Some of the pipelines were just added and no search has been runned yet.",
                "Thus, Pipeline Manager cached only previously seen preprocessors.",
                sep="\n",
            )

            preprocessors_to_save = list(set(cls.pipelines) - set(cls.newly_added))
        else:
            CacheManager.save_preprocessors_to_cache(preprocessors_to_save, cls.best_preprocessor)

    @classmethod
    def add_pipeline(cls, pipeline_description: PipelineDescription):
        new_preprocessors = PreprocessorFactory.create_from_description(pipeline_description)
        for preprocessor in new_preprocessors:
            if preprocessor not in cls.pipelines:
                cls.pipelines.append(preprocessor)
                cls.newly_added.append(preprocessor)

    @classmethod
    def run_search(
        cls,
        np_image: np.ndarray,
        search_strategy_name: str,
        ocr_engine: Optional[OcrEngine] = None,
        cold_start=False,
    ):

        if cold_start:
            pipelines = cls.pipelines
        else:
            pipelines = cls.newly_added

        # Add last-time winner if there is one
        if cls.best_preprocessor:
            pipelines += [cls.best_preprocessor]

        search_strategy = SearchStrategyFactory.create_strategy(search_strategy_name)
        cls.best_preprocessor = search_strategy.search(pipelines, np_image, ocr_engine)

        cls.newly_added = []

    @classmethod
    def get_best_preprocessor(cls):
        if cls.newly_added:
            cls.best_preprocessor = None
            print("New pipelines were added. Do run_search once again to define a new winner")
        return cls.best_preprocessor


class SearchStrategy(ABC):
    @abstractmethod
    def search(
        self, pipelines: list[Preprocessor], np_image: np.ndarray, ocr_engine=None
    ) -> Preprocessor:
        pass


class GridSearch(SearchStrategy):
    """
    Best preprocessor search strategy must have a reference to a
    pipeline manager, and, optionally, to an OCR_engine used for bounding boxes detection
    """

    def search(
        self, pipelines: list[Preprocessor], np_image: np.ndarray, ocr_engine=None
    ) -> Preprocessor:
        competing_images = [preprocessor.process(np_image) for preprocessor in pipelines]
        if ocr_engine:
            competing_images = [ocr_engine.process(np_image) for np_image in competing_images]

        best_image_index = ImageSelector.select_best_image(competing_images)
        assert best_image_index is not None
        return pipelines[best_image_index]


class SearchStrategyFactory:
    """Factory class to create SearchStrategy objects."""

    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> SearchStrategy:
        """Creates a SearchStrategy object based on the given name."""

        strategies = {
            "GridSearch": GridSearch,
            # Add other strategies here (e.g., "RandomizedGridSearch": RandomizedGridSearch)
        }

        if strategy_name not in strategies:
            raise ValueError(f"Invalid search strategy name: {strategy_name}")

        return strategies[strategy_name](**kwargs)


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
    pipeline_manage.load_from_cache()

    # Example Usage
    pipeline1 = PipelineDescription(
        {
            cv2.cvtColor: {"code": [cv2.COLOR_BGR2GRAY]},
            cv2.adaptiveThreshold: {
                "maxValue": [255],
                "adaptiveMethod": [cv2.ADAPTIVE_THRESH_MEAN_C],
                "thresholdType": [cv2.THRESH_BINARY],
                "blockSize": [9, 15],
                "C": [5],
            },
            crop_image: {"minx": [0.1], "maxx": [0.7], "miny": [0.4], "maxy": [0.95]},
            resize_image: {
                "scale_factor": [1, 2],
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
                "blockSize": [31],
                "C": [10, 20, 30, 40],
            },
            crop_image: {"minx": [0.1], "maxx": [0.7], "miny": [0.4], "maxy": [0.95]},
        }
    )

    test_image = cv2.imread("../test_images/test_image1.png")
    pipeline_manage.add_pipeline(pipeline1)
    pipeline_manage.add_pipeline(pipeline2)
    pipeline_manage.run_search(test_image, "GridSearch")
    pipeline_manage.save_to_cache()
    print(pipeline_manage.best_preprocessor)


if __name__ == "__main__":
    main()
