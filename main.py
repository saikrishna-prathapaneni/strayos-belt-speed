import os, sys
PATH = os.path.dirname(os.path.abspath(__file__))
# ZED SDK 4.x
import pyzed.sl as sl
sdk_version = sl.Camera().get_sdk_version() 
if sdk_version.split(".")[0] != "4":
    print("This sample is meant to be used with the SDK v4.x, Aborting.")
    exit(1)

# import other libraries as needed
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt





def handle_depth(z):
    '''
    Parameters:
    z: depth value at pixel location x and y

    Returns:
    Z: valid depth value at the location

    if not present returns 0
    
    '''
    if np.isfinite(z) and not np.isnan(z):
        return z
    return 0


def image_to_world(args, x, y, Z):
    '''
    Parameters:
    x, y: pixel coordinates of the location
    fx,fy: focal length in pixels
    cx,cy: optical center coordinates in pixels
    
    Returns:
    X,Y,Z: world coordinates

    function assumes CCS (camera coordinate system)and WCS (world coordinate system)
    to be aligned on a same origin and axis. 
    ref: https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file-

    '''
    
   
    X =   Z * (x - args.cx)/ (args.fx) 
    Y =   Z * (y - args.cy)/ (args.fy)

    return X, Y, Z


def calculate_median_of_non_outliers(args,data):
    """
    Calculate the median value of a dataset excluding any outliers.
    
    Parameters:
    - data: A list of numerical values representing the dataset.
    
    Returns:
    - The median value of the dataset after removing outliers.
    """
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * args.threshold)
    upper_bound = quartile_3 + (iqr * args.threshold)
    non_outliers = [x for x in data if lower_bound <= x <= upper_bound]
    return np.median(non_outliers)

def is_outlier(args,point, data):
    """
    Determine if a data point is an outlier based on the median and interquartile range.
    
    Parameters:
    - point: The numerical value of the data point to test.
    - data: A list of numerical values representing the dataset.
    
    Returns:
    - A boolean indicating whether the point is an outlier (True) or not (False).
    """
    median = np.median(data)
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * args.threshold)
    upper_bound = quartile_3 + (iqr * args.threshold)
    return point < lower_bound or point > upper_bound


def get_speed(args,
            world_coordinates: list,
            curr_timestamp: float,
            prev_timestamp: float
            ):
    '''
    return speed of conveyor of between frames
    Parameters:
    point1, point2: world coordinates of features identified
    timdelta: difference in time between measuring point1 and point2

    point* formatting: (X*, Y*, Z*)
    Returns:
    speed obtained to travel from point1 to point2
    
    '''
    
    total_dist = 0
    count =0

    for coord in world_coordinates:
        point1 = coord[0]
        point2 = coord[1]
        
        d = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
        total_dist += d
        count +=1
    

    avg_dist = total_dist/count
    print("average distance",avg_dist)
    return avg_dist*60


def extrapolate_keypoints(keypoints, x_offset, y_offset):
        '''
        extrapolate to ICS(Image coordinate system)
        Parameters:
        keypoints: SIFT keypoints
        x_offset: cropped patch x coordinate(pixel location)
        y_offset: cropped patch y coordinate(pixel location)

        Returns:
        extrapolated coordinates for the detected keypoints

        '''
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x_offset, kp.pt[1] + y_offset)
 
        return keypoints

def calculate_dynamic_threshold(args, speeds):
    '''
    Calculates the dynamic speed threshold based standard deviation computed.
    
    Parameters: list of speeds across the frames and thresholds
    
    returns:
    dynamic threshold 
    
    '''
    if len(speeds) < args.adaptive_threshold_window:
        return args.near_zero_base_threshold
    recent_speeds = speeds[-args.adaptive_threshold_window:]
    std_dev = np.std(recent_speeds)
    return max(args.near_zero_base_threshold, std_dev)


def find_features( args,
                    feature_extractor,
                    prev_gray_frame: np.ndarray,
                    current_frame_gray: np.ndarray
                      ):
    
    """
    extracts features(keypoints) using feature extractor for the adjacent frames 
    Parameters:
    feature_extractor: feature extractor based on SIFT or etc..
    frame1, frame2 : grayscale frames passed 
    window: boundaries in which the feature extraction should be constrained
    
    Returns:
    matched features 
    
    """
    prev_keypoints,prev_descriptors,curr_keypoints,curr_descriptors = None, None, None, None
    if args.feature_extractor =="SIFT":
            window_coords = args.SIFT_window_size

            # Extract window coordinates
            (x1, y1), (x2, y2) = window_coords
            prev_window = prev_gray_frame[y1:y2, x1:x2]
            current_window = current_frame_gray[y1:y2, x1:x2]

            prev_keypoints, prev_descriptors = feature_extractor.detectAndCompute(prev_window, None)

            curr_keypoints, curr_descriptors = feature_extractor.detectAndCompute(current_window, None)
            
            # Extrapolate keypoints to the actual size of the image for both frames

            prev_keypoints = extrapolate_keypoints(prev_keypoints,x1, y1)
            curr_keypoints = extrapolate_keypoints(curr_keypoints,x1, y1 )


            # uncomment the following to visualise the cropped images with in which the speed is computed
            # plt.subplot(1, 2, 1)
            # plt.imshow(prev_window, cmap='gray')
            # plt.title('Previous Frame with Keypoints')

            # plt.subplot(1, 2, 2)
            # plt.imshow(current_window, cmap='gray')
            # plt.title('Current Frame with Keypoints')

            # plt.show()
    
    matches= None

    if args.matcher== 'knn':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(prev_descriptors, curr_descriptors, k=2)
        good_matches = []



        for m, n in matches:
            # distance ratio test threshold of 0.7 is set
            if m.distance < args.const* n.distance:
                good_matches.append(m)
        if args.viz:
            matched_image = cv2.drawMatches( prev_gray_frame,
                                             prev_keypoints,
                                             current_frame_gray,
                                             curr_keypoints,
                                             good_matches[:5],
                                             None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(matched_image)
            plt.show()

        return prev_keypoints, curr_keypoints, good_matches
    
    elif args.matcher =="Brute_force":
        pass

def get_world_coordinates(args,
                        prev_keypoints,
                        prev_depth,
                        curr_keypoints,
                        curr_depth,
                        good_matches):
    '''
    get the world coordinates of the selected features
    Parameters:
    prev_keypoints: n-1 frame's keypoints
    prev_depth: n-1 frame's depth
    curr_keypoints: n frame's keypoints
    curr_depth: n frame's depth
    good_matches: filtered mactches between both frames

    Returns:
    world coordinates with respect to origin at Camer Coordinate system(CCS)

    '''
    world_coordinates =[]
    for match in good_matches[:10]:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        p1 = prev_keypoints[idx1].pt
        p2 = curr_keypoints[idx2].pt
        p1 = list(p1)
        p2 = list(p2)

  
        Z1 =  prev_depth[int(p1[1]), int(p1[0])]
        Z2 =  curr_depth[int(p2[1]), int(p2[0])]

        #clean depth values inf and -inf values
        z1 = handle_depth(Z1)
        z2 = handle_depth(Z2)

        p1[0], p1[1], Z1 = image_to_world(args, p1[0], p1[1], z1)

        p2[0], p2[1], Z2 = image_to_world(args, p2[0], p2[1], z2)
        p1.append(Z1)
        p2.append(Z2)
        world_coordinates.append([p1,p2])
    return world_coordinates




def main(args):
    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(os.path.join(PATH, "data", "belt_sample.svo"))
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
        print("Camera Open", status, "Exit program.")
        exit(1)
    
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    timestamp = sl.Timestamp()
    
    sift= None
    if args.feature_extractor =="SIFT":
        sift = cv2.SIFT_create()


    # You may need this to get pixel coordinates from depth coordinates
    camera_parameters = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    print("Camera Parameters: ", camera_parameters.fx, camera_parameters.fy, camera_parameters.cx, camera_parameters.cy)
    
    args.fx = camera_parameters.fx
    args.fy = camera_parameters.fy
    args.cx = camera_parameters.cx
    args.cy = camera_parameters.cy

    key = ''
    fhand = open('results.csv', 'w')
    

    prev_gray_frame = None

    prev_depth = None

    prev_keypoints = None
    
    prev_timestamp = None


   
    speed = 0 # millimeters per second

    speeds = []

    while key != 113:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO reached")
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            print("Error grabbing frame: ", err)
            break

        if prev_gray_frame is not None:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            
            # get depth data of the current frame
            depth_data = depth.get_data()
            # get current frame
            frame = image.get_data()
            
            curr_timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            
            curr_timestamp = curr_timestamp.get_milliseconds()

            #convert to grayscale
            gray_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
      
            
            prev_keypoints, curr_keypoints, good_matches = find_features(args,
                                                                         sift,
                                                                         prev_gray_frame,
                                                                         gray_frame)
            
            world_coordinates = get_world_coordinates(args,
                                                     prev_keypoints,
                                                     prev_depth,
                                                     curr_keypoints,
                                                     depth_data,
                                                     good_matches)

            speed = get_speed(args, world_coordinates, curr_timestamp, prev_timestamp)
            

            dynamic_threshold = calculate_dynamic_threshold(args,speeds)

            # Check for near-zero speed

            if abs(speed) < dynamic_threshold:
                print("Near zero speed detected between frames")
                speed = 0

            # Check if the speed measured is an outlier
            
            if len(speeds) > args.window_size:  # Wait until you have enough data points to calculate the median and IQR
               
                if is_outlier(args, speed, np.array(speeds[-args.window_size:])) and speed!=0 :  # Check the last window_size readings
                    # If it's an outlier, append the median of the non-outliers instead
                    median_speed = calculate_median_of_non_outliers(args, speeds[-args.window_size:])
                    speed = median_speed
                    if args.include_median_speed:
                        speeds.append(median_speed)
                else:
                    speeds.append(speed)
            else:
                speeds.append(speed)


            prev_gray_frame = gray_frame
            prev_depth = depth_data
            prev_timestamp = curr_timestamp

        else:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

            prev_frame = image.get_data()
            prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            prev_timestamp = prev_timestamp.get_milliseconds()
            prev_depth = depth.get_data()

       
        print("Timestamp: ", prev_timestamp, "Speed: ", speed)
        fhand.write(str(prev_timestamp) + "," + str(speed) + "\n")
       

        # Optional: you can display the image with below code
        #cv2.imshow("Image", image.get_data())   
        #key = cv2.waitKey(0)
    
    cam.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get speed of the conveyor")
    parser.add_argument("--svo_file_path",
                         type=str, help="Path to the input SVO file")
    
    parser.add_argument("--feature_extractor",
                        type=str, default="SIFT", help ="pass the feature extractor")
    parser.add_argument("--SIFT_window_size",
                        type=tuple, default=([510,250],[700,400]), help ="window with in which features are extracted")
    parser.add_argument("--matcher",
                        type=str, default="knn", choices=["knn","Brute_force"], help ="feature matching algorithms K nearest neighbours and Brute force")
    parser.add_argument("--viz",
                        type=bool, default=False, help ="visualise the features extracted and mateched frames")
    parser.add_argument("--const",
                        type=float, default=0.7, help =" distance ratio test threshold for good matches")

    parser.add_argument('--threshold',
                         type=float, default=2.0, help='The multiplier for the interquartile range to determine outliers in speed estimation.')
    parser.add_argument('--window_size',
                         type=int, default=150, help='The number of recent readings to consider for outlier detection.')
    parser.add_argument('--include_median_speed',
                         type=bool, default=False, help='Flag to inlcude median value instead of Outlier into the data.')
    parser.add_argument('--adaptive_threshold_window',
                         type=int, default=100, help=' window of values for adaptive threshold, if the frame captures zero or near zero value.')
    parser.add_argument('--near_zero_base_threshold',
                         type=int, default=10, help=' near zero base threshold below which we consider it as a zero.')

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("Script End")
        #print("MSE calculated: ",calculate_mse_for_every_n_samples("results.csv"))
            