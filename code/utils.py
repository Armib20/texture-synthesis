import numpy as np
import random
from scipy import ndimage
from random import randint

def synthRandomPatch(img, tileSize, numTiles, outSize):
    """
    Synthesize texture by randomly tiling patches from source image.
    
    Parameters:
    -----------
    img : ndarray
        Source texture image (grayscale)
    tileSize : int
        Size of each square tile
    numTiles : int
        Number of tiles in each dimension (output will be numTiles x numTiles)
    outSize : int
        Size of the output image (should be numTiles * tileSize)
    
    Returns:
    --------
    ndarray
        Synthesized texture image
    """
    # Get dimensions of the input image
    h, w = img.shape
    
    # Create empty output image
    output = np.zeros((outSize, outSize))
    
    # For each tile position in the output
    for i in range(numTiles):
        for j in range(numTiles):
            # Pick a random location in the source image
            # Ensure we don't go out of bounds
            rand_h = randint(0, h - tileSize)
            rand_w = randint(0, w - tileSize)
            
            # Extract the random patch
            patch = img[rand_h:rand_h+tileSize, rand_w:rand_w+tileSize]
            
            # Place the patch in the output image
            output[i*tileSize:(i+1)*tileSize, j*tileSize:(j+1)*tileSize] = patch
    
    return output

def synthEfrosLeung(img, winsize, outSize, errThreshold=0.1):
    """
    Synthesize texture using Efros and Leung's algorithm.
    
    Parameters:
    -----------
    img : ndarray
        Source texture image (grayscale)
    winsize : int
        Size of the neighborhood window
    outSize : int
        Size of the output image (square)
    errThreshold : float
        Error threshold for selecting best matches
    
    Returns:
    --------
    ndarray
        Synthesized texture image
    """
    # Get dimensions of the input image
    h, w = img.shape
    
    # Create empty output image and a mask to track filled pixels
    output = np.zeros((outSize, outSize))
    mask = np.zeros((outSize, outSize), dtype=bool)
    
    # Place a 3x3 seed patch at the center of the output image
    seed_h = randint(0, h - 3)
    seed_w = randint(0, w - 3)
    seed_patch = img[seed_h:seed_h+3, seed_w:seed_w+3]
    
    center = outSize // 2
    output[center-1:center+2, center-1:center+2] = seed_patch
    mask[center-1:center+2, center-1:center+2] = True
    
    # Half window size for neighborhood calculations
    half_win = winsize // 2
    
    # Create a kernel for dilation to find unfilled pixels with filled neighbors
    kernel = np.ones((3, 3), dtype=bool)
    
    # Continue until all pixels are filled
    while not np.all(mask):
        # Find unfilled pixels with filled neighbors
        dilated = ndimage.binary_dilation(mask, structure=kernel)
        boundary = dilated & ~mask
        
        # Get coordinates of boundary pixels
        boundary_coords = np.where(boundary)
        
        # If no boundary pixels, break (shouldn't happen with proper initialization)
        if len(boundary_coords[0]) == 0:
            break
        
        # Count filled neighbors for each boundary pixel
        filled_neighbors = np.zeros(len(boundary_coords[0]))
        for i in range(len(boundary_coords[0])):
            y, x = boundary_coords[0][i], boundary_coords[1][i]
            # Count filled neighbors in a 3x3 neighborhood
            y_min, y_max = max(0, y-1), min(outSize, y+2)
            x_min, x_max = max(0, x-1), min(outSize, x+2)
            filled_neighbors[i] = np.sum(mask[y_min:y_max, x_min:x_max])
        
        # Sort boundary pixels by number of filled neighbors (descending)
        sorted_indices = np.argsort(-filled_neighbors)
        
        # Process the pixel with the most filled neighbors
        y, x = boundary_coords[0][sorted_indices[0]], boundary_coords[1][sorted_indices[0]]
        
        # Create valid mask for the neighborhood
        y_min, y_max = max(0, y-half_win), min(outSize, y+half_win+1)
        x_min, x_max = max(0, x-half_win), min(outSize, x+half_win+1)
        
        # Extract the neighborhood and its mask
        neighborhood = output[y_min:y_max, x_min:x_max]
        neighborhood_mask = mask[y_min:y_max, x_min:x_max]
        
        # Neighborhood offset from center
        offset_y = half_win - (y - y_min)
        offset_x = half_win - (x - x_min)
        
        # Find the best matching pixel in the input image
        min_ssd = float('inf')
        best_matches = []
        
        # Pad the input image to handle boundary cases
        padded_img = np.pad(img, half_win, mode='reflect')
        
        # Iterate through all possible windows in the input image
        for i in range(half_win, h + half_win):
            for j in range(half_win, w + half_win):
                # Extract window from padded input
                window = padded_img[i-half_win:i+half_win+1, j-half_win:j+half_win+1]
                
                # Adjust window size to match the neighborhood
                win_y_min = half_win - offset_y
                win_y_max = win_y_min + (y_max - y_min)
                win_x_min = half_win - offset_x
                win_x_max = win_x_min + (x_max - x_min)
                
                # Ensure we don't go out of bounds
                if (win_y_min < 0 or win_y_max > winsize or 
                    win_x_min < 0 or win_x_max > winsize):
                    continue
                
                window_crop = window[win_y_min:win_y_max, win_x_min:win_x_max]
                
                # Skip if sizes don't match
                if window_crop.shape != neighborhood.shape:
                    continue
                
                # Calculate SSD only for filled pixels
                diff = (window_crop - neighborhood) ** 2
                ssd = np.sum(diff * neighborhood_mask) / np.sum(neighborhood_mask)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_matches = [(i - half_win, j - half_win)]
                elif ssd <= min_ssd * (1 + errThreshold):
                    best_matches.append((i - half_win, j - half_win))
        
        # Randomly select one of the best matches
        if best_matches:
            best_i, best_j = random.choice(best_matches)
            output[y, x] = img[best_i, best_j]
            mask[y, x] = True
    
    return output
    