# sample code of how to find the tool via hsv thresholding. See README for more info

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([12 - 5,50,50])
upper_hsv = np.array([12 + 5,255,255])
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
res = cv2.bitwise_and(frame,frame, mask= mask)
# ... skip a bit
canny_sector_mask = sector_mask(canny.shape,(model_y_centre,model_x_centre),radius_threshold*model_radius,(0,360))
# ... skip a bit
res[~canny_sector_mask] = 0

tool = ~cv2.subtract(frame,res)
