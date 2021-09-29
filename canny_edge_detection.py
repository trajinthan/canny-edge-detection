#Algorithm for Canny_Edge_Detection

#import necessary libraries
import cv2 as cv
import numpy as np

#load an image
img = cv.imread("sample.jpg")

#STEP 1
#convert image to gray scale
def gray_scale(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return imgGray

gray_scaled_img = gray_scale(img)
cv.imwrite('Gray_scaled_image.jpg',gray_scaled_img)

#STEP 2
#apply gaussian filtering
#generate_gaussian kernel
def gaussian_kernel(kernal_size, sigma=1):
    gaussian_kernal = np.zeros((kernal_size, kernal_size), np.float32)
    i = kernal_size//2
    j = kernal_size//2

    for x in range(-i, i+1):
        for y in range(-j, j+1):
            normal  = 2*np.pi*(sigma**2)
            g = np.exp(-(x**2 + y**2)/(2*sigma**2))
            gaussian_kernal[x+i, y+j] = (1/normal)*g 
    return gaussian_kernal/gaussian_kernal.sum()

def convolution(image, kernal):
    img = image.copy()
    kernal_size = len(kernal)
 
    for k in range(len(image)):
        for i in range(kernal_size//2):
            img[k].insert(0,image[k][-1-(i*2)])
            img[k].append(image[k][1+(i*2)])
    for i in range(kernal_size//2):
        img.insert(0,img[-1-i].copy())
        img.append(image[i].copy())

    image_x = len(img)
    image_y = len(img[0])
    result= []
    kernal_middle = kernal_size//2
    for x in range(kernal_middle, image_x - kernal_middle):
        temp = []
        for y in range(kernal_middle, image_y - kernal_middle):
            value = 0
            for i in range(len(kernal)):
                for j in range(len(kernal)):
                    xn = x + i - kernal_middle
                    yn = y + j - kernal_middle
                    filtered_value = kernal[i][j]*img[xn][yn]
                    value += filtered_value
            temp.append(value)
        result.append(temp)
    return np.array(result)

#generate gaussian kernel of size 7
gaus_kernal = gaussian_kernel(7)
filtered_img = convolution(gray_scaled_img.tolist(),gaus_kernal)
cv.imwrite('Gaussian_filtered_image.jpg',filtered_img)

#STEP 3
#Estimation of "Sobel" gradient strength and direction
def gradient_estimation(img):
    M_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    M_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
   
    G_x = convolution(img.tolist(), M_x.tolist())
    G_y = convolution(img.tolist(), M_y.tolist())

    G = (G_x**2 + G_y**2)**0.5
    G = (G / G.max()) * 255
    theta = np.arctan2(G_y,G_x)

    return (G,theta)

(G,theta) = gradient_estimation(filtered_img)
cv.imwrite('Gradient_estimated_image.jpg',G)

#STEP 4
#non-maximum suppression
def non_max_suppression(img, direction):
    img_height = img.shape[0]
    img_width = img.shape[1]

    new_img = np.zeros(img.shape)
    angle = direction * 180 /np.pi
    # print(angle)
    angle[angle<0] += 180
    # print(angle)
    for i in range(img_height):
        for j in range(img_width):
            try:
                # print(angle[i][j][c])
                before_pixel = 255
                after_pixel = 255
                # link the gradient angle to the pixel direction
                if ( 0 <= angle[i][j] and angle[i][j] < 22.5) or (157.5 <= angle[i][j] and angle[i][j] <=180):
                    q = img[i][j+1]
                    r = img[i][j-1]
                elif (22.5 <= angle[i][j] and angle[i][j] <67.5):
                    q = img[i+1][j-1]
                    r = img[i-1][j+1]
                elif (67.5 <= angle[i][j] and angle[i][j] <112.5):
                    q = img[i+1][j]
                    r = img[i-1][j]
                elif (112.5 <= angle[i][j] and angle[i][j] <157.5):
                    q = img[i-1][j-1]
                    r = img[i+1][j+1]
            
                if (img[i][j] >= q) and (img [i][j] >= r):
                    new_img[i][j] = img[i][j]
                else:
                    new_img[i][j] = 0
            except IndexError:
                pass
    return new_img

suppressed_img = non_max_suppression(G, theta)
cv.imwrite('Suppressed_image.jpg',suppressed_img)

#STEP5
#double threshold
def double_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i,j] >= highThresholdRatio:
                res[i, j] = strong
            elif img[i,j] >= lowThresholdRatio and img[i,j] <= highThresholdRatio:
                res[i, j] = weak
    return res

double_threshold_image = double_threshold(suppressed_img)
cv.imwrite('double_threshold_image.jpg',double_threshold_image.astype(np.uint8))

#hysteresis edge track
def hysteresis(img):
    M, N = img.shape
    size = img.shape

    weak = np.int32(25) 
    strong = np.int32(255)

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

hysteresis_image = hysteresis(double_threshold_image)
cv.imwrite('edge_detected_image.jpg',hysteresis_image)