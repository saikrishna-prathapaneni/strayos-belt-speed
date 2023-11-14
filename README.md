# strayos-belt-speed


## Overview
This script measures the speed of a conveyor belt carrying debris using a 760p60fps ZED camera. It utilizes the ZED SDK, OpenCV, NumPy, and Matplotlib to process video frames and calculate the speed of objects moving on the conveyor belt.

Sample images (Frame captured on left camera and corresponding depth)
<p align="center">
  <img src="assets/image12.png"  width="200" height="300" />
  <img src="assets/image7.png"  width="200" height="300" />
</p>


## Procedure:
The script follows these steps to compute the speed of the conveyor belt:

1. **Initialization**: The script starts by initializing the ZED camera using the ZED SDK and loads video frames sequentially. It sets up the camera parameters and checks if the correct version of the SDK is being used.

2. **Feature Extraction**: Using the Scale-Invariant Feature Transform (SIFT) method or a specified alternative, the script extracts features and corresponding keypoints from the video frames. The features represent distinct points or characteristics in the image of the conveyor belt.

      **Note:** To improve the computation speed and reduce noise in detected features only a patch of image is considered based on `--SIFT_window_size` parameter that is default to be `([510,250],[700,400]) `.

3. **Feature Matching**: Features from consecutive frames are matched to track the movement of points across frames. This process uses the KNN (k-Nearest Neighbors).

    **Note:**  A parameter `--const` is used for distance ratio test threshold for good matches (Refer to const vs MAE graph below) to select best `--const` value, Brute Force Extraction will be implemented in the further development.

4. **Coordinate Transformation**: The script then transforms image coordinates to world coordinates. This step is crucial for calculating the actual distance moved by points on the conveyor belt in the real world.

   **Note:** For reliable estimate all the three coordinates (X, Y , Z) are considered for displacement measurement.

6. **Speed Calculation**: The speed of the conveyor belt is calculated based on the displacement of matched points in world coordinates and the time elapsed between frames.

   **Note:** For speed estimate a constant interval (of 1/60 sec) is considered between frames.

7. **Outlier Detection and Noise Removal**: To improve accuracy, the script includes outlier detection. It uses the interquartile range method to identify and exclude any abnormal speed readings that might skew the results.

   **Note:** The parameters `--window_size` (default: 100) is used to find the number of recent readings to consider for outlier detection and the parameter `--threshold` for adjusting the outlier sensitivity.

8. **Result Output**: Finally, the script outputs the calculated speed of the conveyor belt for each frame, along with the corresponding timestamp, and saves this data to a CSV file.




<br>
Following are the speeds captured after removing outliers through `--window_size` of 30, 100, 200 and No outlier removal.
<br>
Mean speeds Generated for different`--window_size` are `1708 mm/sec for 30, 1645 mm/sec for 100, 1675 mm/sec for 200 and over 1798 mm/sec for no outlier removal.` 
<br><br>
<p align="center">
  <img src="assets/image8.png"  width="200" height="200" />
  <img src="assets/image5.png"  width="200" height="200" />
  <img src="assets/image3.png"  width="200" height="200" />
  <img src="assets/image2.png"  width="200" height="200" />
</p>


Following is the graph between MAE (Mean Absolute Error) evalated for 10 frames assuming true speed to be 1600 and finalised on the `--const` parameter to be 0.7.
<br>
![MAE vs Frame count ](assets/mae_vs_frame_count.png)
<br>
## Prerequisites
- Python 3.8
- ZED SDK 4.x
- OpenCV (`cv2`) library
- NumPy (`numpy`) library
- Matplotlib (`matplotlib`) library
- A 760p60fps ZED camera

## Installation
1. Ensure Python 3.8 is installed on your system.
2. Install the ZED SDK version 4.x from the [StereoLabs website](https://www.stereolabs.com/developers/).
3. Install the required Python libraries:
```pip install opencv-python numpy matplotlib```


## Usage
Run the script using Python:
`python main.py [OPTIONS]`
The script will start processing the video feed from the recorded SVO video file and measure the speed of the conveyor belt for each frame.

## The script accepts several command-line arguments for customization:

--feature_extractor: Feature extractor to use (default is SIFT(Scale Invariant Feature Transform)).<br>
--SIFT_window_size: Window size within which features are extracted.<br>
--matcher: Feature matching algorithm (KNN or Brute-force).<br>
--viz: Enable visualization of feature extraction and matching.<br>
--const: Distance ratio test threshold for good matches.<br>
--threshold: Multiplier for the interquartile range for outlier detection.<br>
--window_size: Number of recent readings to consider for outlier detection.<br>
--include_median_speed: Flag to include median value instead of outlier into the data.<br>

## Further considerations and Future Development

1. **KALMAN filter** - In the implementation to handle noise, a computationally low intensive IQR method is employed to further improve the noise handling Kalman filter is under development for accurate speed estimation
2. **Exponential Moving average** - To improve speed of the algorithm simpler EMA can be used to reduce the outlier.
3. **Block Matching** - To improve the accuracy of the features, sliding window algorithm can be employed for the matching of selected blocks in consecutive blocks.
4. **Iterative Closest Point (ICP)** - Iterative closest point algorithm to compute to alignment between the point clouds of two adjacent frames can be used to compute the speed, yet a computationally intensive procedure but with a better estmate of speed.
