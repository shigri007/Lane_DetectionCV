import cv2
import numpy as np

def canny(img):
    """Applies the Canny edge detection algorithm."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges
def region_of_interest(img):
    """Applies an image mask to keep only the region of interest."""
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([
        [(200, height), (800, 350), (1200, height)]
    ], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def hough_lines(img):
    """Detects lines in the image using the Hough Transform."""
    return cv2.HoughLinesP(img, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)

def add_weighted(frame, line_img):
    """Combines the original frame with the line image."""
    return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

def display_lines(img, lines):
    """Draws lines on an image."""
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_img

def make_points(image, line):
    """Converts a line's slope and intercept into endpoints."""
    height = image.shape[0]
    slope, intercept = line
    y1 = height
    y2 = int(height * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    """Averages the slopes and intercepts of lines to produce a single line for each side."""
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    left_line = make_points(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.average(right_fit, axis=0)) if right_fit else None
    
    return [left_line, right_line] if left_line and right_line else None

def main():
    cap = cv2.VideoCapture("roadVideo.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        canny_image = canny(frame)
        if canny_image is None:
            break
        
        cropped_canny = region_of_interest(canny_image)
        lines = hough_lines(cropped_canny)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = add_weighted(frame, line_image)
        
        cv2.imshow("Result", combo_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
