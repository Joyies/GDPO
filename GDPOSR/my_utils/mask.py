import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_complexity_degree(img, weight=1.0):
    try:
        # img = (img + 1.0) / 2.0 # (0, 1)
        h, w = img.shape
        # img = cv2.resize(img, dsize=(256,256))
        img = cv2.resize(img, dsize=(64,64))
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) # (-4, 4)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2) # [-4*2**0.5, 4*2**0.5]
        hist, _ = np.histogram(edge_magnitude, bins=512, range=(-5.5, 5.5))
        hist = hist / (hist.sum()+1e-5)
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) 

        complexity_degree = entropy**weight*h*w
    except Exception as e:
        print(f"img: {img.shape}")
    return complexity_degree

def create_complexity_matrix(gray_img, patch_size=60):
    """
    Divide a grayscale image into patches and calculate the complexity of each patch 
    to generate a complexity matrix.

    Parameters:
        gray_img: Input grayscale image (NumPy array).
        patch_size: Size of each patch (default: 6x6).
        
    Returns:
        complexity_matrix: The complexity matrix (same size as the input image).
    """

    h, w = gray_img.shape
    complexity_matrix = np.zeros((h, w))
    
    rows = h // patch_size
    cols = w // patch_size
    
    for i in range(rows):
        for j in range(cols):
            patch = gray_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            
            complexity = calculate_complexity_degree(patch)
            
            complexity_matrix[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = complexity
    
    if rows * patch_size < h:
        for j in range(cols):
            patch = gray_img[rows*patch_size:, j*patch_size:(j+1)*patch_size]
            complexity = calculate_complexity_degree(patch)
            complexity_matrix[rows*patch_size:, j*patch_size:(j+1)*patch_size] = complexity
    
    if cols * patch_size < w:
        for i in range(rows):
            patch = gray_img[i*patch_size:(i+1)*patch_size, cols*patch_size:]
            complexity = calculate_complexity_degree(patch)
            complexity_matrix[i*patch_size:(i+1)*patch_size, cols*patch_size:] = complexity
    
    if rows * patch_size < h and cols * patch_size < w:
        patch = gray_img[rows*patch_size:, cols*patch_size:]
        complexity = calculate_complexity_degree(patch)
        complexity_matrix[rows*patch_size:, cols*patch_size:] = complexity
    
    return complexity_matrix

def binarize_complexity_matrix(complexity_matrix, threshold=50):
    """
    Binarize the complexity matrix.

    Parameters:
        complexity_matrix: The complexity matrix.
        threshold: The threshold value. Elements greater than this value are set to 1, 
                and elements less than or equal to it are set to 0.
                
    Returns:
        binary_matrix: The binarized matrix.
        fidelity_zero_ratio: The proportion of the fidelity region (ratio of zeros).
        detail_one_ratio: The proportion of the detail region (ratio of ones).
    """    
    binary_matrix = np.zeros_like(complexity_matrix, dtype=np.uint8)
    binary_matrix[complexity_matrix > threshold] = 1
    
    unique_values = np.unique(binary_matrix)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError("The input matrix must be a binary matrix containing only 0 and 1.")
    
    total_elements = binary_matrix.size
    
    zero_count = np.count_nonzero(binary_matrix == 0)
    one_count = np.count_nonzero(binary_matrix == 1)
    
    fedilty_zero_ratio = round((zero_count / total_elements), 2)
    detail_one_ratio = round((one_count / total_elements), 2)

    return binary_matrix, fedilty_zero_ratio, detail_one_ratio

def extract_and_dilate_edges(gray, threshold1=100, threshold2=200, dilation_size=3, downscale_factor=8):
    
    edges = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    h, w = dilated_edges.shape
    new_h, new_w = h // downscale_factor, w // downscale_factor
    downsampled_edges = cv2.resize(dilated_edges, (new_w, new_h), interpolation=cv2.INTER_AREA)
    downsampled_edges = downsampled_edges/255.0
    
    _, downsampled_edges_mask = cv2.threshold(downsampled_edges, 0.498, 1.0, cv2.THRESH_BINARY)

    return downsampled_edges_mask

if __name__ == '__main__':
    img = cv2.imread("DrealSR/test_HR/DSC_1412_x1.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img/255

    complexity_matrix = create_complexity_matrix(gray_img, patch_size=10)
    print(complexity_matrix.shape)
    binary_matrix, fedilty_zero_ratio, detail_one_ratio = binarize_complexity_matrix(complexity_matrix, threshold=50)
    print(fedilty_zero_ratio, detail_one_ratio)
    print(binary_matrix.shape)