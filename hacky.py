import numpy as np
import cv2
from predict_math import *
print("Loaded libraries")



input_video = r"D:\Documents\Eye stuff\OpenCV\vid_008.avi"
#input_video = r"D:\Documents\Eye stuff\OpenCV\PGY4 cataract_Sikder_Houle_chop_WilmerOR6_11-28-2011.avi"
output_video = r"D:\Documents\Eye stuff\OpenCV\processed_video.avi"
output_directory = "D:\\Documents\\Eye stuff\\OpenCV\processed\\"

past_frames = []
past_circles = []
past_lines = []

# state variables
hsv_colour_thresholding = 12 #color mask to isolate the pupil, should be found via user input
hsv_variance = 5
frame_num = 50
minRadius=85    #circle
maxRadius=105   #circle
param2=8       #HoughCircles
lower=70        #Canny
upper=115       #Canny
radius_threshold = 1.13      #sector mask
best_x_circles = 10
circumference_hugger = 23
angle_variance = .7
y_int_variance = 20

cap = cv2.VideoCapture(input_video)

# grab metadata about video
dimensions = cap.read()[1].shape[:-1]

# go back to the first frame
cap.set(1, 0)

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
out = cv2.VideoWriter(output_video, fourcc, 20.0, dimensions)

#holds information about a frame that is used in predictive functions
class circle_container:
    def __init__(self, centre_x, centre_y, radius):
        self.centre_x=centre_x
        self.centre_y=centre_y
        self.radius=radius

class line_container:
    def __init__(self, slope, intercept):
        self.slope=slope
        self.intercept=intercept
        
class frame_container:
    def __init__(self,  circle, line):
        self.circle=circle
        self.line=line

def user_input_calibration():
  global minRadius, maxRadius, param2, lower, upper
  global fourcc, cap, dimensions, input_video, frame_num

  print("Preparing first frame for parameter selection")

  # create trackbars
  def nothing(x):pass
  cv2.namedWindow('image')
  cv2.createTrackbar('lower','image',10,200,nothing)
  cv2.createTrackbar('upper','image',10,200,nothing)


  while(1):
      ret, frame = cap.read()
      cap.set(1, frame_num)

      lower = cv2.getTrackbarPos('lower','image')
      upper = cv2.getTrackbarPos('upper','image')

      image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      image = cv2.medianBlur(image, 7)
      image = cv2.Canny(image, lower, upper)

      circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,
                                 minDist=100,
                                 param1=1,
                                 param2=1,
                                 minRadius=50,
                                 maxRadius=200)
      image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
      
      try:
          model_x_centre,model_y_centre,model_radius = circles[0][0]
          cv2.circle(image,(model_x_centre,model_y_centre),model_radius,(255,255,0), 2)
          cv2.circle(frame,(model_x_centre,model_y_centre),model_radius,(255,255,0), 2)
      except:pass

      cv2.imshow('image', frame)
   
      if cv2.waitKey(1) >= 30:break

  cv2.destroyAllWindows()
  minRadius = int(.9*model_radius)
  maxRadius = int(1.1*model_radius)

  print("Canny bounds: {}, {}".format(lower,upper))
  print("Radius bounded around {}".format(int(model_radius)))

  # sieve for param2
  param2 = 1
  ret, frame = cap.read()
  cap.set(1, 0)
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  image = cv2.medianBlur(image, 7)
  image = cv2.Canny(image, lower, upper)
  while cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,
             minDist=100,
             param1=1,
             param2=param2,
             minRadius=minRadius,
             maxRadius=maxRadius) != None:param2+=1
  param2/=2
  print("param2 = {}".format(param2))

#function definitions
def circle_distance(x,y,r,xx,yy,rr):
  return np.sqrt((x+r-xx-rr)**2+(y-yy)**2) + np.sqrt((x-r-xx++rr)**2+(y-yy)**2) + np.sqrt((x-xx)**2+(y+r-rr-yy)**2) + np.sqrt((x-xx)**2+(y-r+rr-yy)**2)

def cart2pol(x,y,Xc,Yc):
    rho = np.sqrt((x-Xc)**2 + (y-Yc)**2)
    phi = np.arctan2(y-Yc, x-Xc)
    return(rho, phi)

def pol2cart(rho, phi, Xc,Yc):
    x = rho * np.cos(phi)+Xc
    y = rho * np.sin(phi)+Yc
    return(x, y)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def sector_mask(shape,centre,radius,angle_range):
    """
    Boolean mask for determining if a point is inside a circle
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    while( tmax < tmin):
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

#user_input_calibration()
frame_num=00
cap.set(1, frame_num)
while 1:
    print "="*10+"%d"%frame_num+"="*10
    pause=False #failure in reading current frame
    ret, frame = cap.read()

    #hsv thresholding to remove green
    blur = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([hsv_colour_thresholding-hsv_variance,50,50])
    upper_hsv = np.array([hsv_colour_thresholding+hsv_variance,255,255])
    
    # Threshold the HSV image to get only the range of colors
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # Bitwise-AND mask and original image

    #OTSU thresholding to reveal tip
    image_OTSU = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #image_OTSU = cv2.medianBlur(image_OTSU, 3) #blur to remove a bit of noise
    (thresh, image_OTSU) = cv2.threshold(image_OTSU, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #canny edge detection on the thresholded figure
    canny = cv2.Canny(cv2.medianBlur(mask, 3), lower, upper)
    #second canny of OTSU for extended line detection
    canny_OTSU = cv2.Canny(cv2.medianBlur(image_OTSU, 3), lower, upper)


    #convert to bgr as a quick fix to display output
    #copy them all, but in RGB instead of greyscale
    image_OTSU_graphics = cv2.cvtColor(image_OTSU, cv2.COLOR_GRAY2BGR)
    canny_graphics = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    canny_OTSU_graphics = cv2.cvtColor(canny_OTSU, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1,
                 minDist=1,
                 param1=1,
                 param2=param2,
                 minRadius=minRadius,
                 maxRadius=maxRadius)
    if circles is not None:
         circles = circles[0][:best_x_circles] #select x best circles, irregardless of any of their dimensions

    (predicted_cx, predicted_cy, predicted_r) = (frame_prediction(past_circles, "centre_x"),
                                                frame_prediction(past_circles, "centre_y"), 
                                                frame_prediction(past_circles, "radius"))

     #if the predictor actually gave a prediction...
    if predicted_cx is not None:
      model_x_centre,model_y_centre,model_radius=sorted(circles, 
                      key=lambda x:circle_distance(x[0],x[1],x[2], predicted_cx, predicted_cy, predicted_r))[0]
                      
      #if the best fit has a "distance" of over 50 from what we would expect, assume there was a bad fit
      if circle_distance(model_x_centre,model_y_centre,model_radius, predicted_cx, predicted_cy, predicted_r) > 100:
        pause=True

    #if the predictor gave a null result, then just select the first fit (ie use default ranking)    
    else:
      model_x_centre,model_y_centre,model_radius=circles[0]
      
    if pause == False:
      
      past_circles.append(circle_container(model_x_centre, model_y_centre, model_radius))
         
      #drawing
      cv2.circle(frame,(model_x_centre,model_y_centre),model_radius,(0,0,255), 2)
      """
      cv2.circle(image_OTSU_graphics,(model_x_centre,model_y_centre),model_radius,(0,0,255), 2)
      cv2.circle(canny_graphics,(model_x_centre,model_y_centre),model_radius,(0,0,255), 2)
      cv2.circle(frame,(model_x_centre,model_y_centre),model_radius,(0,0,255), 2)
      cv2.circle(canny_OTSU_graphics,(model_x_centre,model_y_centre),model_radius,(0,0,255), 2)
      """ 
    
      
    else:pause=True #this is from:::: if circles is not None:

    
    if not pause:
        #mask out the parts of the image far outside the eye
        canny_sector_mask = sector_mask(canny.shape,(model_y_centre,model_x_centre),radius_threshold*model_radius,(0,360))
        image_OTSU_graphics[~canny_sector_mask] = 0
        canny_graphics[~canny_sector_mask] = 0
        canny_OTSU_graphics[~canny_sector_mask] = 0
        
        
        image_OTSU[~canny_sector_mask] = 0
        canny[~canny_sector_mask] = 0
        canny_OTSU[~canny_sector_mask] = 0

        plines = cv2.HoughLinesP(canny,
                        rho=1, theta=np.pi/360/3,
                        threshold=30,
                        minLineLength=30,
                        maxLineGap=10)
                        
        plines2 = cv2.HoughLinesP(canny_OTSU,
                        rho=1, theta=np.pi/360/3,
                        threshold=30,
                        minLineLength=30,
                        maxLineGap=10)
        
        #find contours of the OTSU image
        tool = np.zeros(dimensions, np.uint8)
        second_canvas = np.zeros(dimensions, np.uint8)
        _, contours, _ = cv2.findContours(canny_OTSU,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        min_area = 5
        
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        # draw contours
        cv2.drawContours(tool, large_contours, -1, 255, thickness=10)        
        
        
        #remove the line segments that hug the circumference of the eye too closely
        if plines is not None:
            plines = [i[0] for i in plines]
            plines = [l for l in plines if circumference_hugger<np.abs(cart2pol(l[0],l[1], model_x_centre, model_y_centre)[0]-model_radius) + np.abs(cart2pol(l[2],l[3], model_x_centre, model_y_centre)[0]-model_radius)]

        if plines2 is not None:
            plines2 = [i[0] for i in plines2]
            plines2 = [l for l in plines2 if circumference_hugger<np.abs(cart2pol(l[0],l[1], model_x_centre, model_y_centre)[0]-model_radius) + np.abs(cart2pol(l[2],l[3], model_x_centre, model_y_centre)[0]-model_radius)]
        
        #combines the two sets of lines
        if plines is None and plines2 is not None:
            plines = plines2
        elif plines is not None and plines2 is not None:
            plines+=plines2
        
        if plines is not None:
            angles = [np.arctan((l[1]-l[3])/(l[0]-l[2]*1.00001)+.00001) for l in plines]
            final_angles=[]
            final_intercepts=[]
            p_y_int, p_slope = (frame_prediction(past_lines, "magic_wall", 2),
                 frame_prediction(past_lines, "angle", 2))

        #draw the lines
        for l in plines:
            #slope
            slope = (l[1]-l[3])/(l[0]-l[2]*1.00001)+.00001 #dealing w/ inf
            angle = np.arctan(slope)
            if abs(np.median(angles)-slope) > 2 * np.std(angles):
                continue
            
            y_int = l[0] + slope * (model_x_centre-50-l[1])            
            
            if p_slope is not None:
                if abs(angle-p_slope) > angle_variance or abs(p_y_int - y_int) > y_int_variance:
                    print "SKIPPED\n%3f\t%d"%(angle,y_int)
                    continue
                
            print "%3f\t%d"%(angle,y_int)
            
            final_angles.append(angle)
            final_intercepts.append(y_int)
            
            # green for line segments
            """
            cv2.line(image_OTSU_graphics, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
            cv2.line(canny_graphics, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
            cv2.line(frame, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
            cv2.line(canny_OTSU_graphics, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
            """
            
            #not display, but processing
            cv2.line(second_canvas, (l[0]-int(5/slope),l[1]-5), (l[2],l[3]), 255, 6)
            cv2.circle(second_canvas, (l[0],l[1]) ,10 ,255)
            
        if(len(final_angles)) > 0:
            past_lines.append(line_container(np.median(final_angles), np.median(final_intercepts)))


        #if a tool is detected
        if len(plines) > 0: 
            tool = cv2.bitwise_and(~cv2.medianBlur(image_OTSU, 7), second_canvas)
            a,b = np.nonzero(tool)
            #sanity check
            if len(a) > 0:
                x=min(b)
                i = np.where(b == x)[0][0]
                y=a[i]
                tip=(x+2,y+2)
                frame = cv2.add(cv2.bitwise_and(frame,frame, mask=cv2.bitwise_not(tool)) , cv2.cvtColor(tool, cv2.COLOR_GRAY2BGR))
        
                cv2.circle(frame, tip ,2 ,(255, 255,0), 2)
        else:
            tool=np.zeros(dimensions, np.uint8)
    
    

    cv2.imshow('frame', frame)
    """
    cv2.imshow('OTSU', image_OTSU_graphics)
    cv2.imshow('canny hsv', canny_graphics)
    cv2.imshow('canny OTSU', canny_OTSU_graphics)
    cv2.imshow('TOOL', tool)
    """
    
    
    cv2.imwrite(output_directory + "%d.jpg" % frame_num, frame)
    frame_num+=1

    if pause and cv2.waitKey(10000) >= 30: #pause at each failure
      print "BREAK FAILURE BREAK"
      break 
    if cv2.waitKey(20) >= 30:break
      



cap.release()
out.release()
cv2.destroyAllWindows()

