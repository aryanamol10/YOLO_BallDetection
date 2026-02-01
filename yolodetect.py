from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

model = YOLO("PyTorch_model.pt")

ball_num = 0
#calibrated value
focal_length = 1755
ball_w = 0.4925
IMG_WIDTH = 1280 
CX = IMG_WIDTH / 2

conversion_factor = 0.3048
detected_objects = []

def rundistcheck(x1, x2):

    #First, we need to find the focal length
    w_pixels = x2 - x1

    
    centerx = (x1+x2)/2
    distance_feet = 2

    dz = (ball_w*focal_length)/w_pixels

    dx = ((centerx - CX)*dz)/focal_length

    detected_objects.append((dz, dx))



def checkdetect(ball_num, x1, x2):
    ball_num+=1
    if (ball_num-1)>1:
        print("multiple balls")
    elif (ball_num-1)==1:
        print("Only one ball")
    elif (ball_num)-1==0:
        print("zero balls")
    

def feet2m(feet):
    return feet*0.3048

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

current_x1 = 0
current_x2 = 0

while True:
    # 2. Capture a frame frame-by-frame
    ret, frame = cap.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 3. Convert the captured color frame to grayscale
    # OpenCV uses BGR color space by default
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_3_channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    gray_3_channel = frame
    results = model.predict(source=gray_3_channel, show=False)

    names = results[0].names

    for result in results:
        boxes = result.boxes # Boxes object
        for box in boxes:
            # Extract coordinates in xyxy format
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            rundistcheck(x1, x2)

            # Extract confidence and class ID
            confidence = box.conf[0]
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            pixels = (x2 - x1)+(y2-y1)
            # Create the label text
            #label = f"Conf: {confidence:.2f} | {feet2m(380/pixels):.2f} | {pixels:.2f}m"

            # Define color for the bounding box (BGR format)
            color = (0, 255, 0) # Green color

            # Draw the rectangle (bounding box)
            cv2.rectangle(gray_3_channel, (x1, y1), (x2, y2), color, 2)

            # Put the label text above the bounding box
            #cv2.putText(gray_3_channel, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ball_num+=1

        if len(detected_objects)>=2:
            #easy to check through
            b1 = detected_objects[0]
            b2 = detected_objects[1]

            #distance formula
            gap_ft = math.sqrt((b2[0] - b1[0])**2 + (b2[1] - b1[1])**2)

            gap_ft_zero = gap_ft<=0.5
            if(gap_ft_zero):
                gap_ft = 0
                
            gap_m = gap_ft * conversion_factor

        
            #print(f"Gap: {gap_ft:.2f} ft | {gap_m:.2f} m")
            cv2.putText(frame, f"Dist: {gap_ft:.2f}ft | Dist: {gap_m:.2f}m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        #checkdetect(ball_num, current_x1, current_x2)
        detected_objects.clear()
        ball_num = 0
    cv2.imshow("detections", gray_3_channel)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()