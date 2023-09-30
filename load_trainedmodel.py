import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tensorflow.keras.models import load_model



window_val =[]

class_names = ['normal', 'abnormal']
threshold = 0.5  # Set a threshold value, change this to whatever value you see fit

# It can be used to reconstruct the model identically.
#model = keras.models.load_model("3d_image_classification.h5")
model = load_model('3d_image_classification.h5')
print("Loaded model from disk")
# summarize model.
model.summary()


def read_mhd_file(filepath):
    """Read and load volume"""
    # Read file
    scan = sitk.ReadImage(filepath)
    scan = sitk.GetArrayFromImage(scan)
    scan = np.moveaxis(scan, 0, 2)
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = np.flip(img , axis=2)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_mhd_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    #mem()
    return volume



image = process_scan('/home/mustafa/project/LUNA16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize[2]):
        for y in range(0, image.shape[1], stepSize[1]):
            for x in range(0, image.shape[0], stepSize[0]):
                # yield the current window
                yield (x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]])

# parameters
window_size = (31, 31, 31)  # The size of the scanning window, change to suit your needs
step_size = (5, 5, 5)  # The amount of pixels the window moves at each step, change to suit your needs


# slide the window over the image
for (x, y, z, window) in sliding_window(image, step_size, window_size):
    # Here you can apply your 3D CNN to the window, e.g.:
    if window.shape[0] != window_size[0] or window.shape[1] != window_size[1] or window.shape[2] != window_size[2]:
        continue  # Skip if window doesn't meet size requirements (at the edges)

    window = np.expand_dims(window, axis=0)  # Add an extra dimension for the batch size
    window = np.expand_dims(window, axis=4)  # Add an extra dimension for grayscale
    prediction = model.predict(window)  # Apply your model to the window
    # Convert the prediction to a readable format
    #prediction_dict = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    prediction_dict = {'normal': 1 - prediction[0][0], 'abnormal': prediction[0][0]}

    # Print only if the probability for 'abnormal' is higher than the threshold
    if prediction_dict['abnormal'] > threshold:
        print(f'High probability of abnormality at window position {(x, y, z)}: {prediction_dict["abnormal"]}')
        #print(f'Prediction for window at position {(x, y, z)}: {prediction}')
