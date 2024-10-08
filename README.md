# Auto-Preprocessing for Computer Vision

This Python module helps you automate the selection of the best preprocessing pipeline for your computer vision tasks. It uses a combination of parameter grid search, interactive visualization, and caching to efficiently find and apply optimal preprocessing steps to your images.

## Features

- **Define Multiple Pipelines:** Create and store multiple preprocessing pipelines, each with different sequences of cv2 functions and a range of parameter values.
- **Grid Search:**  The module performs an exhaustive grid search across all defined pipelines and parameter combinations.
- **Interactive Visualization:** A user-friendly image selector allows you to visually compare the results of different preprocessing steps and choose the best one.
- **Caching:**  The module caches the results of the search process, so repeated runs with the same pipelines and image will be much faster.
- **Preprocessor Construction:** Once the best pipeline is selected, the module automatically constructs a `Preprocessor` object that you can use to apply the optimal preprocessing to new images.

## Installation

1. **Git Clone:** Clone this repository into your working directory:

   ```bash
   git clone https://github.com/Kalmy8/auto_preprocessing
   ```

2. **Requirements:** Make sure you have the required libraries installed:

   ```bash
   pip install -r requirements.txt
   ```

Alternatively:

1. **Install as package using pip:**
   ```bash
   pip install prepCV
   ```
   
## Usage
1. **Define Pipelines:** Create `PipelineDescription` objects to define your preprocessing pipelines.

   ```python
   import cv2
   from prepCV import PipelineDescription, PipelineManager

   pipeline1 = PipelineDescription({
       cv2.cvtColor: {'code': [cv2.COLOR_BGR2GRAY]},
       cv2.adaptiveThreshold: {
           'maxValue': [255],
           'adaptiveMethod': [cv2.ADAPTIVE_THRESH_MEAN_C], 
           # ... other parameters ...
       },
       # ... other cv2 functions ...
   })

   pipeline2 = PipelineDescription({
       # ... define another pipeline ...
   })
   ```

2. **Add Pipelines to Manager:** 

   ```python
   pipeline_manage = PipelineManager()
   pipeline_manage.add_pipeline(pipeline1)
   pipeline_manage.add_pipeline(pipeline2) 
   ```

3. **Run Search:**

   ```python
   # Load your image
   image = cv2.imread('your_image.jpg')

   # Run the search and select the best preprocessor
   pipeline_manage.run_search(image, 'GridSearch') 

   # Get the best preprocessor
   best_preprocessor = pipeline_manage.get_best_preprocessor()
   ```

3. **Save Search Results to cache:**

```python
   from prepCV import CacheManager
   
   # Restore previous search result 
   pipeline_manage = PipelineManager()
   pipeline_manage.load_from_cache()

   # Degine some new pipeline
      pipeline3 = PipelineDescription({
       # ... define another pipeline ...
   })
   
   # Add pipelines to PipelineManager
   pipeline_manage.add_pipeline(pipeline1)
   pipeline_manage.add_pipeline(pipeline2)
   pipeline_manage.add_pipeline(pipeline3)
   
   # This search will only compare new pipeline3 to previous best seen pipeline
   pipeline_manage.run_search(image, 'GridSearch') 

   # Save results to cache
   pipeline_manage.save_to_cache()
```
   
5. **Use the Preprocessor:**

   ```python
   processed_image = best_preprocessor.process(new_image)
   # Now you can use the `processed_image` for your tasks. 
   ```


