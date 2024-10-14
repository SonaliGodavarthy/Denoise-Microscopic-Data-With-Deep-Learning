import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import median_filter
from tifffile import imread



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
        #denoised_image = denoise_image_with_wiener(noisy_img)
        denoised_channel = wiener(0, mysize=(5, 5))
        denoised_channel.append(denoised_channel)
        denoised_image = np.stack(denoised_channel, axis=-1)
        denoised_image = denoised_image.astype('uint8')
    elif filter_num == 5:
        denoised_image = median_filter(noisy_img, size=3)
    return denoised_image


def plot_results(original_image, noisy_image, denoised_image, PSNR1, PSNR2,filter_num =1):
    # Calculate SNR
    Title = ""
    SNR = calculate_snr(original_image)
    SNR1 = calculate_snr(noisy_image)
    SNR2 = calculate_snr(denoised_image)

    # Calculate PSNR
    # rangePSNR = np.max(original_image) - np.min(original_image)
    # PSNR1 = calculate_psnr(original_image, noisy_image, rangePSNR)
    # PSNR2 = calculate_psnr(original_image, denoised_image, rangePSNR)

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
    axs[0][0].imshow(original_image, cmap="magma")
    axs[0][0].title.set_text(f'Original Image - SNR = {SNR:.2f}')
    axs[0][1].imshow(noisy_image, cmap="magma")
    axs[0][1].title.set_text(f'Noisy Image - SNR = {SNR1:.2f} ; PSNR = {PSNR1:.2f}')
    axs[0][2].imshow(denoised_image, cmap="magma")
    axs[0][2].title.set_text(f'Denoised Image - SNR = {SNR2:.2f} ; PSNR = {PSNR2:.2f}')

    # Visualize zoomed in image
    axs[1][0].imshow(original_image[100:200,150:250], cmap="magma")
    axs[1][0].title.set_text(f'Original Image')
    axs[1][1].imshow(noisy_image[100:200,150:250], cmap="magma")
    axs[1][1].title.set_text(f'Noisy Image')
    axs[1][2].imshow(denoised_image[100:200,150:250], cmap="magma")
    axs[1][2].title.set_text(f'Denoised Image')
    plt.show()


# Read image into an array
#addr = 'images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif'
addr = 'C:/Users/Quratulain Naqvi/PycharmProjects/RAML Project/DecoNoising/Mouse actin/diaphragm.tif'
img = imread(addr)[:,:512,:512]
print(img.shape)

#subtract mean value for the background
img = img - 520

img_GT = np.mean(img[:,...], axis =0)[np.newaxis,...]

psnr_result = []
psnr_input = []

# We iterate over all test images.
for index in range(img.shape[0]):
    im = img[index]
    gt = img_GT[0]  # The ground truth is the same for all images

    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    # Here we are predicting the deconvolved and denoised image
    denoised_img = apply_filter(im, 1)
    # calculate PSNR
    rangePSNR = np.max(gt) - np.min(gt)
    psnr_result.append(calculate_psnr(gt, denoised_img, rangePSNR))
    psnr_input.append(calculate_psnr(gt, im, rangePSNR))
    print("image:", index)
    print("PSNR input", calculate_psnr(gt, im, rangePSNR))
    print("PSNR denoised", calculate_psnr(gt, denoised_img, rangePSNR))
    print('-----------------------------------')

# We display the results for the last test image
vmi = np.percentile(gt, 0.01)
vma = np.percentile(gt, 99)

# plt.figure(figsize=(15, 15))
# plt.subplot(1, 3, 1)
# plt.title('Noisy image')
# plt.imshow(im, cmap="magma")
#
#
# plt.subplot(1, 3, 2)
# plt.title('Denoised output')
# plt.imshow(denoised_img, cmap="magma")
#
# plt.subplot(1, 3, 3)
# plt.title('Ground Truth')
# plt.imshow(gt, cmap="magma")
#
# plt.figure(figsize=(15, 15))
# plt.subplot(1, 3, 1)
# plt.title('Noisy image')
# plt.imshow(im[100:200, 150:250], cmap="magma")
#
# plt.subplot(1, 3, 2)
# plt.title('Denoised output')
# plt.imshow(denoised_img[100:200, 150:250], cmap="magma")
#
# plt.subplot(1, 3, 3)
# plt.title('Ground Truth')
# plt.imshow(gt[100:200, 150:250], cmap="magma")
#

avg_psnr_input1 = np.mean(np.array(psnr_input))
avg_psnr_denoised = np.mean(np.array(psnr_result))

print("Avg PSNR input:", np.mean(np.array(psnr_input)), '+-(2SEM)',
      2 * np.std(np.array(psnr_input)) / np.sqrt(float(len(psnr_input))))
print("Avg PSNR denoised:", np.mean(np.array(psnr_result)), '+-(2SEM)',
      2 * np.std(np.array(psnr_result)) / np.sqrt(float(len(psnr_result))))

# plt.show()


# denoise the image using box filter
# 1. Box filter
# 2. Gaussian filter
# 3. Median filter
# 4. Wiener filter
# 5. Box3D filter
#denoised_img = apply_filter(noisy_g,2)
plot_results(gt, im, denoised_img, avg_psnr_input1, avg_psnr_denoised,1)




