from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)

FPS = 120
cap.set(cv2.CAP_PROP_FPS, FPS)

ret, frame = cap.read()

model = YOLO("PyTorch_model.pt")

#calibrated value
focal_length = 1755
ball_w = 0.04
IMG_WIDTH = 1280 
CX = IMG_WIDTH / 2


detected_objects = [(0,0)]



def runspeedcheck(x1, y1):

    detected_objects.insert(0, (x1, y1))

        #easy to check through
    b1 = detected_objects[0]
    b2 = detected_objects[1]

        #distance formula
    distance = math.sqrt((b2[0] - b1[0])**2 + (b2[1] - b1[1])**2)
    time = 1000/FPS

    detected_objects.pop(-1)

    speed = distance/time

    if(speed <= 0.2):
        return 0
    else:
        return speed
    

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

            # Extract confidence and class ID
            confidence = box.conf[0]
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            # Create the label text
            #label = f"Conf: {confidence:.2f} | {feet2m(380/pixels):.2f} | {pixels:.2f}m"

            # Define color for the bounding box (BGR format)
            color = (0, 255, 0) # Green color

            # Draw the rectangle (bounding box)
            cv2.rectangle(gray_3_channel, (x1, y1), (x2, y2), color, 2)

            # Put the label text above the bounding box
            #cv2.putText(gray_3_channel, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
            #print(f"Gap: {gap_ft:.2f} ft | {gap_m:.2f} m")
            cv2.putText(frame, f"Dist: {runspeedcheck(x1, y1):.2f}m/s ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        #checkdetect(ball_num, current_x1, current_x2)
    cv2.imshow("detections", gray_3_channel)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()