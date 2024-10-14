import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import median_filter


# Add Gaussian noise
def add_gaussian_noise(image, mean=0, var=1000):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype('uint8')


# Add Salt n pepper noise
def add_salt_n_pepper_noise(img):
    rows = img.shape[0]
    cols = img.shape[1]

    # adding noise to image
    x_pixels = np.random.randint(0, rows, 5000)
    y_pixels = np.random.randint(0, cols, 5000)

    noisy_image = img.copy()
    noisy_image[x_pixels[:2500], y_pixels[:2500]] = 255
    noisy_image[x_pixels[2500:], y_pixels[2500:]] = 0
    return noisy_image


def denoise_image_with_wiener(noisy_image):
    # Applying Wiener filter
    denoised_channels = []
    for i in range(3):  # Assuming a 3-channel (RGB) image
        channel = noisy_image[:, :, i]
        denoised_channel = wiener(channel, mysize=(5, 5))
        denoised_channels.append(denoised_channel)
    denoised_image = np.stack(denoised_channels, axis=-1)
    return denoised_image.astype('uint8')



def calculate_snr(image):

    # Ensure the image is in numpy array format
    image = np.asarray(image)

    # If the image has multiple channels (e.g., RGB), convert it to grayscale
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Calculate the mean pixel value (signal)
    signal = np.mean(image)

    # Calculate the standard deviation of the pixel values (noise)
    noise = np.std(image)

    # Calculate the SNR
    snr = signal / noise

    return snr


def calculate_psnr(gt, pred, range_=255.0):
        mse = np.mean((gt - pred) ** 2)
        return 20 * np.log10((range_) / np.sqrt(mse))


def apply_filter(noisy_img, filter_num=1):
    if filter_num == 1:
        denoised_image = cv.blur(noisy_img, (3, 3))
    elif filter_num == 2:
        denoised_image = cv.GaussianBlur(noisy_img, (5,5), 0)
    elif filter_num == 3:
        denoised_image = cv.medianBlur(noisy_img, 3)
    elif filter_num == 4:
        denoised_image = denoise_image_with_wiener(noisy_img)
    elif filter_num == 5:
        denoised_image = median_filter(noisy_img, size=3)
    return denoised_image


def plot_results(original_image, noisy_image, denoised_image,filter_num =1):
    # Calculate SNR
    Title = ""
    SNR = calculate_snr(original_image)
    SNR1 = calculate_snr(noisy_image)
    SNR2 = calculate_snr(denoised_image)

    # Calculate PSNR
    rangePSNR = np.max(original_image) - np.min(original_image)
    PSNR1 = calculate_psnr(original_image, noisy_image, rangePSNR)
    PSNR2 = calculate_psnr(original_image, denoised_image, rangePSNR)

    if (filter_num==1):
        Title = "Box Filter"
    elif (filter_num==2):
        Title = "Gaussian Filter"
    elif (filter_num==3):
        Title = "Median Filter"
    elif (filter_num==4):
        Title = "Wiener Filter"
    elif (filter_num==5):
        Title = "Box3D Filter"

    # visualize actual image vs noisy image vs denoised image
    fig, axs = plt.subplots(2, 3, figsize=(25, 20))
    plt.suptitle(Title)
    axs[0][0].imshow(original_image)
    axs[0][0].title.set_text(f'Original Image - SNR = {SNR:.2f}')
    axs[0][1].imshow(noisy_image)
    axs[0][1].title.set_text(f'Noisy Image - SNR = {SNR1:.2f} ; PSNR = {PSNR1:.2f}')
    axs[0][2].imshow(denoised_image)
    axs[0][2].title.set_text(f'Denoised Image - SNR = {SNR2:.2f} ; PSNR = {PSNR2:.2f}')

    # Visualize zoomed in image
    axs[1][0].imshow(original_image[100:200,150:250])
    axs[1][0].title.set_text(f'Original Image')
    axs[1][1].imshow(noisy_image[100:200,150:250])
    axs[1][1].title.set_text(f'Noisy Image')
    axs[1][2].imshow(denoised_image[100:200,150:250])
    axs[1][2].title.set_text(f'Denoised Image')
    plt.show()


# Read image into an array
#addr = 'images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'
addr = 'images/Img_new.tiff'
img = cv.imread(addr)

noisy_g = add_gaussian_noise(img)
cv.imwrite('new_noisy_image.tiff',noisy_g)
noisy_s = add_salt_n_pepper_noise(img)

# denoise the image using box filter
# 1. Box filter
# 2. Gaussian filter
# 3. Median filter
# 4. Wiener filter
# 5. Box3D filter
denoised_img = apply_filter(noisy_g,2)
plot_results(img, noisy_g, denoised_img,2)




