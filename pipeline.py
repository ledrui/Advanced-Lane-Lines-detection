import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
# %matplotlib inline

# Apply a threshold on the sobel magnitude
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Draw a mask on top of an image
def add_binary_mask(img, m):
    m2 = np.zeros_like(img)
    m2[:, :, 0] = m*255
    m2[:, :, 1] = m
    m2[:, :, 2] = m
    img = np.where(m2, m2, img)
    return img

# Class for perspective transforms
class PerspectiveTransformer():
    def __init__(self, src, dist):
        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    # Apply perspective transform
    def warp(self, img):
        return cv2.warpPerspective(img, self.Mpersp, (img.shape[1], img.shape[0]))

    # Reverse perspective transform
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]))

# Applies the HLS and sobel masks to the image
def mask_image(img):
    img = img.copy()

    # Apply a mask on HLS colour channels
    # This selects pixels with higher than 100 saturation and lower than 100 hue
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(hls[:, :, 0])
    mask[(hls[:, :, 2] > 100) & (hls[:, :, 0] < 100)] = 1

    # Apply a sobel magnitude threshold
    # I apply a more lenient mag_thresh to the upper part of the transformed image, as this part is blurrier
    # and will therefore have smoother gradients.
    # On the bottom half, this selects pixels with >10 sobel magnitude, and on the top half,
    # selects pixels with >35 sobel magnitude
    upper_mag = mag_thresh(img, 3, (10, 255))
    lower_mag = mag_thresh(img, 3, (35, 255))

    mag_mask = np.zeros_like(lower_mag)
    mag_mask[:int(mag_mask.shape[0]/2), :] = upper_mag[:int(mag_mask.shape[0]/2), :]
    mag_mask[int(mag_mask.shape[0]/2):, :] = lower_mag[int(mag_mask.shape[0]/2):, :]

    # Use the bitwise OR mask of both masks for the final mask
    final_mask = np.maximum(mag_mask, mask)

    # Return the transformed mask
    return final_mask

# Find the peaks of the bottom half, for sliding window analysis
def find_initial_peaks(final_mask, bottom_pct=0.5):
    # bottom_pct: How much of the bottom to use for initial tracer placement

    shape = final_mask.shape

    bottom_sect = final_mask[-int(bottom_pct*shape[0]):, :]

    left_peak = bottom_sect[:, :int(0.5*shape[1])].sum(axis=0).argmax()
    right_peak = bottom_sect[:, int(0.5*shape[1]):].sum(axis=0).argmax() + 0.5*shape[1]

    # Return x-position of the two peaks
    return left_peak, right_peak

# This applies the sliding window approach to find lane pixels, and then fits a polynomial to the found pixels.
def sliding_window_poly(final_mask, left_peak, right_peak, num_chunks=10, leeway=80):
    # num_chunks: Number of chunks to split sliding window into
    # leeway: Number of pixels on each side horizontally to consider

    # Split the image vertically into chunks, for analysis.
    chunks = []
    assert final_mask.shape[0] % num_chunks == 0, 'Number of chunks must be a factor of vertical resolution!'
    px = final_mask.shape[0] / num_chunks # Pixels per chunk
    for i in range(num_chunks):
        chunk = final_mask[i*px:(i+1)*px, :]
        chunks.append(chunk)

    # Reverse the order of the chunks, in order to work from the bottom up
    chunks = chunks[::-1]

    # Loop over chunks, finding the lane centre within the leeway.
    lefts = [left_peak]
    rights = [right_peak]

    left_px, left_py, right_px, right_py = [], [], [], []

    for i, chunk in enumerate(chunks):
        offset = (num_chunks-i-1)*px

        last_left = int(lefts[-1])
        last_right = int(rights[-1])

        # Only consider pixels within +-leeway of last chunk location
        temp_left_chunk = chunk.copy()
        temp_left_chunk[:, :last_left-leeway] = 0
        temp_left_chunk[:, last_left+leeway:] = 0

        temp_right_chunk = chunk.copy()
        temp_right_chunk[:, :last_right-leeway] = 0
        temp_right_chunk[:, last_right+leeway:] = 0

        # Save the x, y pixel indexes for calculating the polynomial
        left_px.append(temp_left_chunk.nonzero()[1])
        left_py.append(temp_left_chunk.nonzero()[0] + offset)

        right_px.append(temp_right_chunk.nonzero()[1])
        right_py.append(temp_right_chunk.nonzero()[0] + offset)

    # Create x and y indice arrays for both lines
    left_px = np.concatenate(left_px)
    left_py = np.concatenate(left_py)
    right_px = np.concatenate(right_px)
    right_py = np.concatenate(right_py)

    # Fit the polynomials!
    l_poly = np.polyfit(left_py, left_px, 2)
    r_poly = np.polyfit(right_py, right_px, 2)

    return l_poly, r_poly

# Calculate the lane line curvature
def get_curvature(poly, mask):
    yscale = 30 / 720 # Real world metres per y pixel
    xscale = 3.7 / 700 # Real world metres per x pixel

    # Convert polynomial to set of points for refitting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
    fitx = poly[0] * ploty ** 2 + poly[1] * ploty + poly[2]

    # Fit new polynomial
    fit_cr = np.polyfit(ploty * yscale, fitx * xscale, 2)

    # Calculate curve radius
    curverad = ((1 + (2 * fit_cr[0] * np.max(ploty) * yscale + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curverad

# Plot the polygons on the image
def plot_poly_orig(fitl, fitr, orig):
    # Draw lines from polynomials
    ploty = np.linspace(0, orig.shape[0]-1, orig.shape[0])
    fitl = fitl[0]*ploty**2 + fitl[1]*ploty + fitl[2]
    fitr = fitr[0]*ploty**2 + fitr[1]*ploty + fitr[2]

    pts_left = np.array([np.transpose(np.vstack([fitl, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fitr, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Create an overlay from the lane lines
    overlay = np.zeros_like(orig).astype(np.uint8)
    cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))

    # Apply inverse transform to the overlay to plot it on the original road
    overlay = transform.unwarp(overlay)

    # Add the overlay to the original unwarped image
    result = cv2.addWeighted(orig, 1, overlay, 0.3, 0)
    return result

# Find the offset of the car and the base of the lane lines
def find_offset(l_poly, r_poly):
    lane_width = 3.7  # metres
    h = 720  # height of image (index of image bottom)
    w = 1280 # width of image

    # Find the bottom pixel of the lane lines
    l_px = l_poly[0] * h ** 2 + l_poly[1] * h + l_poly[2]
    r_px = r_poly[0] * h ** 2 + r_poly[1] * h + r_poly[2]

    # Find the number of pixels per real metre
    scale = lane_width / np.abs(l_px - r_px)

    # Find the midpoint
    midpoint = np.mean([l_px, r_px])

    # Find the offset from the centre of the frame, and then multiply by scale
    offset = (w/2 - midpoint) * scale
    return offset

# Buffer for retaining curvature and polygon information between frames
last_rad = None
last_l_poly = None
last_r_poly = None

# Function to apply to frames of video
def process_frame(img):
    global last_rad, last_l_poly, last_r_poly

    # Define weights for smoothing
    rad_alpha = 0.05
    poly_alpha = 0.2

    # Undistort the image using the camera calibration
    img = calibration.undistort(img)

    # Keep the untransformed image for later
    orig = img.copy()

    # Apply perspective transform to the image
    img = transform.warp(img)

    # Apply the HLS/Sobel mask to detect lane pixels
    mask = mask_image(img)

    # Find initial histogram peaks
    left_peak, right_peak = find_initial_peaks(mask)

    # Get the sliding window polynomials for each line line
    l_poly, r_poly = sliding_window_poly(mask, left_peak, right_peak, leeway=80)

    # Update polynomials using weighted average with last frame
    if last_l_poly is None:
        # If first frame, initialise buffer
        last_l_poly = l_poly
        last_r_poly = r_poly
    else:
        # Otherwise, update buffer
        l_poly = (1 - poly_alpha) * last_l_poly + poly_alpha * l_poly
        r_poly = (1 - poly_alpha) * last_r_poly + poly_alpha * r_poly
        last_l_poly = l_poly
        last_r_poly = r_poly

    # Calculate the lane curvature radius
    l_rad = get_curvature(l_poly, mask)
    r_rad = get_curvature(r_poly, mask)

    # Get mean of curvatures
    rad = np.mean([l_rad, r_rad])

    # Update curvature using weighted average with last frame
    if last_rad is None:
        last_rad = rad
    else:
        last_rad = (1 - rad_alpha) * last_rad + rad_alpha * rad

    # Create image
    final = plot_poly_orig(l_poly, r_poly, orig)

    # Write radius on image
    cv2.putText(final, 'Lane Radius: {}m'.format(int(last_rad)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)

    # Write lane offset on image
    offset = find_offset(l_poly, r_poly)
    cv2.putText(final, 'Lane Offset: {}m'.format(round(offset, 4)), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)

    return final

# Initialise the camera calibration, so it can be applied to future images
from calibration   import CameraCalibration
calib_imgs = [mpimg.imread(f) for f in sorted(glob.glob('./camera_cal/*.jpg'))]
calibration = CameraCalibration(calib_imgs, 9, 6)

src = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]]).astype(np.float32)
dst = np.array([[320, 0], [320, 720], [960, 720], [960, 0]]).astype(np.float32)

# Create transformer object, this means that the transformer matrix only needs to be computed once
transform = PerspectiveTransformer(src, dst)

from moviepy.editor import VideoFileClip

white_output = 'out11.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!

white_clip.write_videofile(white_output, audio=False)
