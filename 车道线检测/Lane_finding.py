# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)

import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""    
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)       
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def endpoint(line_x):
    line_ = []
    line_y = []
    if line_x:
        for line in line_x:
            line_.append([line[0], line[1]])
            line_.append([line[2], line[3]])
            line_y.append(line[1])
            line_y.append(line[3])
        max_index = line_y.index(max(line_y))
        min_index = line_y.index(min(line_y))
        max_point = line_[max_index]
        min_point = line_[min_index]
    else:
        max_point = [0, 0]
        min_point = [0, 0]
    return [max_point, min_point]


def symmetry_Point(point, normal):
    '''
    X = m
    point_ = (2m-x,y)
    '''
    point_ = [2 * normal - point[0], point[1]]
    return point_


def cross_point(line_left, line_right, left_xy, right_xy):
    print(f'cross_point: {left_xy, right_xy}')
    print(f'cross_point: {line_left, line_right}')

    if not line_left[0][1]:
        x1 = left_xy[0][0]
        y1 = left_xy[0][1]
        x2 = left_xy[1][0]
        y2 = left_xy[1][1]
    else:
        x1 = line_left[0][0]
        y1 = line_left[0][1]
        x2 = line_left[1][0]
        y2 = line_left[1][1]

    if not line_right[0][1]:
        x3 = right_xy[0][0]
        y3 = right_xy[0][1]
        x4 = right_xy[1][0]
        y4 = right_xy[1][1]

    else:
        x3 = line_right[0][0]
        y3 = line_right[0][1]
        x4 = line_right[1][0]
        y4 = line_right[1][1]




    print(y2, y1, x2, x1)
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    print(x,y)

    return [int(x), int(y)]

def draw_lines(img, lines, left_xy, right_xy, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    img_y, img_x, _ = img.shape
    print(img_x, img_y)
    line_right = []
    line_left = []
    #print(f'lines: {lines}')
    for line in lines:
        #print(f'line: {line}')
        x1, y1, x2, y2 = line
        #for x1,y1,x2,y2 in line:
        print((y2-y1)/(x2-x1))
        if (y2-y1)/(x2-x1) < 0:
            line_left.append([x1,y1,x2,y2])
        elif (y2-y1)/(x2-x1) > 0:
            line_right.append([x1,y1,x2,y2])
    print(line_left, line_right)
    #lines_endpoint = []

    endpoint_left_ = endpoint(line_left)
    endpoint_right_ = endpoint(line_right)

    endpoint_left_c = endpoint_left_
    endpoint_right_c = endpoint_right_
    print(endpoint_left_, endpoint_right_)
    #print(any(endpoint_left), any(endpoint_right))
    endpoint_left = []
    endpoint_right = []
    print((endpoint_left_[0][0] and endpoint_left_[1][0]))

    if not endpoint_left_[0][0] or not endpoint_right_[0][0]:

        normal = int(img_x/2)
        if not endpoint_left_[0][0]:
            endpoint_left_c = []
            symmetry_Point_ = symmetry_Point(endpoint_right_[0], normal)
            endpoint_left_c.append(symmetry_Point_)
            symmetry_Point_ = symmetry_Point(endpoint_right_[1], normal)
            endpoint_left_c.append(symmetry_Point_)
        elif not endpoint_right_[0][0]:
            endpoint_right_c = []
            symmetry_Point_ = symmetry_Point(endpoint_left_[0], normal)
            endpoint_right_c.append(symmetry_Point_)
            symmetry_Point_ = symmetry_Point(endpoint_left_[1], normal)
            endpoint_right_c.append(symmetry_Point_)

    cross_point_ = cross_point(endpoint_left_c, endpoint_right_c, left_xy, right_xy)
    cross_y = cross_point_[1]
    endpoint_y = cross_y * 1.1

    if endpoint_left_[0][0] and endpoint_left_[1][0]:

        if endpoint_left_[0][1] != img_y:
            #(endpoint_left_[0][0] - x)  * (540 - endpoint_left_[1][1]) = (endpoint_left_[1][0] - x) * (540 - endpoint_left_[0][1])
            x = (endpoint_left_[1][0] * (img_y - endpoint_left_[0][1]) - endpoint_left_[0][0] * (
                    img_y - endpoint_left_[1][1])) / (endpoint_left_[1][1] - endpoint_left_[0][1])

            ii = [int(x + 0.5), img_y - 1]
        else:
            ii = endpoint_left_[0]
        endpoint_left.append(ii)


        if endpoint_left_[1][1] != int(endpoint_y):
            x = (endpoint_left_[1][0] * (endpoint_y - endpoint_left_[0][1]) - endpoint_left_[0][0] * (
                        endpoint_y - endpoint_left_[1][1])) / (endpoint_left_[1][1] - endpoint_left_[0][1])

            ii = [int(x + 0.5), int(endpoint_y)]
        else:
            ii = endpoint_left_[1]
        endpoint_left.append(ii)
        #left_xy = endpoint_left

    else:
        endpoint_left = left_xy


    if endpoint_right_[0][0] and endpoint_right_[1][0]:

        if endpoint_right_[0][1] != img_y:
            ii = [int(endpoint_right_[0][0] * img_y / endpoint_right_[0][1] + 0.5), img_y]
        else:
            ii = endpoint_right_[0]
        endpoint_right.append(ii)

        if endpoint_right_[1][1] != int(endpoint_y):
            ii = [int(endpoint_right_[1][0] * endpoint_y / endpoint_right_[1][1] + 0.5), int(endpoint_y)]
        else:
            ii = endpoint_right_[1]
        endpoint_right.append(ii)
        #right_xy = endpoint_right

    else:
        endpoint_right = right_xy


    '''
    if any(endpoint_left) and any(endpoint_right):
        print(endpoint_left[0])
        if endpoint_left[1][1] < endpoint_right[1][1]:
            endpoint_right = []
            endpoint_right.append([490 - endpoint_left[0][0] + 490, endpoint_left[0][1]])
            endpoint_right.append([490 - endpoint_left[1][0] + 490, endpoint_left[1][1]])
        else:
            endpoint_left = []
            endpoint_left.append([490 - (endpoint_right[0][0] - 490), endpoint_right[0][1]])
            endpoint_left.append([490 - (endpoint_right[1][0] - 490), endpoint_right[1][1]])
    elif any(endpoint_left):
        endpoint_right = []
        endpoint_right.append([490 - endpoint_left[0][0] + 490, endpoint_left[0][1]])
        endpoint_right.append([490 - endpoint_left[1][0] + 490, endpoint_left[1][1]])
    elif any(endpoint_right):
        endpoint_left = []
        endpoint_left.append([490 - (endpoint_right[0][0] - 490), endpoint_right[0][1]])
        endpoint_left.append([490 - (endpoint_right[1][0] - 490), endpoint_right[1][1]])'''
    #lines_endpoint.append(endpoint_left)
    #lines_endpoint.append(endpoint_right)
    #if any(endpoint_left) and any(endpoint_right):
    print(endpoint_left, endpoint_right)
    #line_left, line_right
    cv2.line(img, (endpoint_left[0][0], endpoint_left[0][1]), (endpoint_left[1][0], endpoint_left[1][1]), color, thickness)
    cv2.line(img, (endpoint_right[1][0], endpoint_right[1][1]), (endpoint_right[0][0], endpoint_right[0][1]), color, thickness)
    #print(f'lines_endpoint: {len(lines_endpoint)}')
    '''for line in lines:
        print(f'line.shape: {line.shape} ')
        for x1,y1,x2,y2 in line:

            cv2.line(img, (x1, y1), (x2, y2), color, thickness)'''
    left_xy = endpoint_left
    right_xy = endpoint_right
    return left_xy, right_xy

def outlier_detection(lines_slope):
    #print(lines_slope)
    median = np.median(np.abs(lines_slope))
    print(median)
    b = 1#.4826  # 这个值应该是看需求加的，有点类似加大波动范围之类的
    mad = b * np.median(np.abs(np.abs(lines_slope) - median))
    print(f'mad: {mad}')
    lower_limit = median - (1.4 * mad)
    upper_limit = median + (1.4 * mad)
    print(lower_limit, upper_limit)
    index_ = []
    for index, item in enumerate(lines_slope):
        if np.abs(item) < 0.5 or np.abs(item) > 0.8:#np.abs(item) < lower_limit or np.abs(item) > upper_limit or
            index_.append(index)

    return index_

def slope(lines):
    slope_ = []
    print(lines)
    for line in lines:
        #print(f'line.shape: {line.shape} ')
        x1, y1, x2, y2 = line
        _ = (y2 - y1) / (x2 - x1)
        slope_.append(_)
    return slope_

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, left_xy, right_xy):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    print(lines)
    #print(lines.any())
    if lines is None:
        lines = []
        _ = left_xy[0]
        _.extend(left_xy[1])

        lines.append(_)
        _ = right_xy[0]
        _.extend(right_xy[1])
        lines.append(_)

    else:
        lines_ = []
        print(lines)
        for line in lines:
            # print(f'line.shape: {line.shape} ')
            for x1, y1, x2, y2 in line:
                lines_.append([x1, y1, x2, y2])
        lines = lines_

    lines_slope = slope(lines)
    print(lines_slope, lines)
    outlier_index = outlier_detection(lines_slope)
    print(outlier_index)
    lines_ = []
    for index, item in enumerate(lines):
        #print(item, index)
        if index not in outlier_index:
            lines_.append(item)
    lines = lines_
    print(f'lines: {lines} {np.shape(lines)}')
    #for
    #remove_ixs = np.where(lines > 400)#[0]
    #print(f'remove_ixs: {remove_ixs} {np.shape(remove_ixs)}')
    #lines_ = np.delete(lines, 5, 0)
    #print(f'lines_x: {lines_x} {np.shape(lines_x)}')
    #lines = lines.tolist()
    #print(f'lines: {lines} {np.shape(lines)}')

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #plt.imshow(line_img)
    left_xy, right_xy = draw_lines(line_img, lines, left_xy, right_xy)
    #plt.imshow(line_img)
    return line_img, left_xy, right_xy

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os

def test_image(img, left_xy, right_xy):

    imshape = img.shape
    print(imshape)
    img_y, img_x, _ = imshape
    image = np.copy(img)
    
    img = grayscale(img)

    plt.imshow(img)
    plt.show()
    
    kernel_size = 1
    img = gaussian_blur(img, kernel_size)
    
    plt.imshow(img)
    plt.show()
    
    low_threshold = 150
    high_threshold = 222
    img = canny(img, low_threshold, high_threshold)
    #masked_edges = np.copy(img)
       
    plt.imshow(img)
    plt.show()
    '''        
    vertices = np.array([[(imshape[0]*0.3,imshape[0]*0.9),(int(img_x/2*0.9), img_y*0.65), (int(img_x/2*1.1), img_y*0.65), (imshape[1]*0.90,imshape[0]*0.9)]], dtype=np.int32)
    img = region_of_interest(img, vertices)'''

    '''
    img =
    
    plt.imshow(img)
    plt.show()'''
    #img_y, img_x


    rho = int(img_y * img_x * 0.000010) # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = int(img_y * img_x * 0.00010) # 150    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = int(img_y * img_x * 0.00007) #minimum number of pixels making up a line
    max_line_gap = int(img_y * img_x * 0.0002) # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on


    img, left_xy, right_xy = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, left_xy,
                                               right_xy)
    '''
    rho = int(img_y * img_x * 0.000009) # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = int(img_y * img_x * 0.00016) # 150    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = int(img_y * img_x * 0.00006) #minimum number of pixels making up a line
    max_line_gap = int(img_y * img_x * 0.0002) # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on  


    img_long, left_xy, right_xy = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, left_xy,
                                               right_xy)'''

    plt.imshow(img)
    plt.show()
        
    #initial_img = np.dstack((masked_edges, masked_edges, masked_edges))
    img = weighted_img(img, image, α=0.8, β=1., γ=0.)

    plt.imshow(img)
    plt.show()
    return img, left_xy, right_xy


left_xy = [[0, 0], [0, 0]]
right_xy = [[0, 0], [0, 0]]

image_path = 'test_images/'
image_list = os.listdir(image_path)
image_path_out = 'test_images_output/'
for image in image_list:
    image_ = mpimg.imread(image_path + image)
    test_, left_xy, right_xy = test_image(image_, left_xy, right_xy)
    mpimg.imsave(image_path_out + image, test_)
    #cv2.imwrite(image_path_out + image, test_)

#reading in an image

image = mpimg.imread('test_images_output/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)
#plt.show()

# Import everything needed to edit/save/watch video clips

from moviepy.editor import VideoFileClip

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    global left_xy, right_xy
    result, left_xy, right_xy = test_image(image, left_xy, right_xy)
    # you should return the final output (image where lines are drawn on lanes)
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


yellow_output = 'test_videos_output/solidYellowLeft.mp4'

clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)


challenge_output = 'test_videos_output/challenge.mp4'


clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)'''