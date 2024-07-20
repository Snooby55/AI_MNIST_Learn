
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")

# Creating a 28 x 28 white canvas
width, height = 28, 28
canvas = np.ones((height, width, 3), dtype=np.uint8) # * 255

# Setting the window size
window_name = "Draw"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, width * 15, height * 15)

# Check box to track mouse clicks
drawing = False
last_x = None
last_y = None


# Функція для малювання
def draw(event, x, y, flags, param):
    global drawing, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x = x
        last_y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a line without blurring
            #cv2.line(canvas, (last_x, last_y), (x, y), (0, 0, 0), 1)
            # Draw a line with a blur
            cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.GaussianBlur(canvas, (5, 5), 0)

            last_x = x
            last_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(canvas, (0, 0), (28, 28), (0, 0, 0), -1)

# A function that checks whether all pixels in an image are black.
def is_all_black(canvas):
  # Convert image to RGB
  canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

  # Check if all RGB channels are black
  return np.all(canvas == [0, 0, 0])


def pre_process_image(img):
    # Resize to model's input size (modify if needed)
    img = cv2.resize(img, (28, 28))

    # Convert to grayscale and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    # Reshape to match model's input format (optional)
    img = img.reshape((1, 28, 28, 1))  # Add channel dimension if needed

    return img


# Function for prediction
def predict():
    global canvas

    # Get a copy of the image for prediction
    img_copy = canvas.copy()

    # Image preprocessing
    preprocessed_img = pre_process_image(img_copy)

    # Make a predict using our model
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)

    # Display the predict on the console
    print(f"Prediction: {predicted_class}")


# Binding functions to mouse events
cv2.setMouseCallback(window_name, draw)


# A loop to display the canvas and update it on changes
while True:
    # Update image
    prev_img = canvas.copy()
    cv2.imshow(window_name, canvas)

    # Check the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Only update the window if the image has actually changed
    # to reduce CPU usage
    if not np.array_equal(prev_img, canvas):
        if not is_all_black(canvas):
            predict()
        cv2.waitKey(1)  # Small delay to avoid overwhelming CPU

# Closing the window and releasing resources
cv2.destroyAllWindows()
