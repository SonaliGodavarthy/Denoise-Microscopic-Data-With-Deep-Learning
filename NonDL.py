import cv2
import numpy as np
from scipy.signal import wiener

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv2.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype('uint8')

def compute_snr(original, noisy, denoised):
    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)
    denoised = denoised.astype(np.float64)
    signal_power = np.mean(original ** 2)
    noise = original - denoised
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def denoise_image_with_box_filter(image, kernel_size=3):
    denoised_image = cv2.blur(image, (kernel_size, kernel_size))
    return denoised_image

def denoise_image_with_gaussian_filter(image,kernel_size=3):
    denoised_image = cv2.GaussianBlur(image, (kernel_size,kernel_size), 0)
    return denoised_image

def denoise_image_with_median_filter(image,kernel_size=3):
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

def denoise_image_with_box3d_filter(image, kernel_size=3):
    denoised = cv2.boxFilter(image, -1, (kernel_size,kernel_size))
    return denoised

def denoise_image_with_wiener_filter(image, kernel_size=3):
    denoised_channels = []
    for i in range(3):
        channel = image[:, :, i]
        denoised_channel = wiener(channel, mysize=(kernel_size, kernel_size))
        denoised_channels.append(denoised_channel)
    denoised_image = np.stack(denoised_channels, axis=-1)
    return denoised_image.astype('uint8')

image = cv2.imread('images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif')

noisy_image = add_gaussian_noise(image, mean=0, var=100)
denoised_box_filter = denoise_image_with_box_filter(noisy_image)
denoised_gaussian_filter = denoise_image_with_gaussian_filter(noisy_image)
denoised_median_filter = denoise_image_with_median_filter(noisy_image)
denoised_box3d_filter = denoise_image_with_box3d_filter(noisy_image)
denoised_wiener_filter = denoise_image_with_wiener_filter(noisy_image)

print("SNR (Box Filter):", compute_snr(image, noisy_image, denoised_box_filter))
print("SNR (Gaussian Filter):", compute_snr(image, noisy_image, denoised_gaussian_filter))
print("SNR (Median Filter):", compute_snr(image, noisy_image, denoised_median_filter))
print("SNR (Box3D Filter):", compute_snr(image, noisy_image, denoised_box3d_filter))
print("SNR (Wiener Filter):", compute_snr(image, noisy_image, denoised_wiener_filter))

cv2.imshow('Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image (Box Filter)', denoised_box_filter)
cv2.imshow('Denoised Image (Gaussian Filter)', denoised_gaussian_filter)
cv2.imshow('Denoised Image (Median Filter)', denoised_median_filter)
cv2.imshow('Denoised Image (Box3D Filter)', denoised_box3d_filter)
cv2.imshow('Denoised Image (Wiener Filter)', denoised_wiener_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()
