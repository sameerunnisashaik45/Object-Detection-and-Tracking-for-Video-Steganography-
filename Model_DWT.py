import numpy as np
import pywt
import cv2


# DWT decomposition function
def dwt_decompose(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH


# DWT reconstruction function
def dwt_reconstruct(LL, LH, HL, HH):
    coeffs = (LL, (LH, HL, HH))
    img_reconstructed = pywt.idwt2(coeffs, 'haar')
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)


# Function to embed secret image into cover image
def embed_images(cover_img, secret_img, alpha=0.1):
    # Step 1: DWT decomposition of cover image
    cover_LL, cover_LH, cover_HL, cover_HH = dwt_decompose(cover_img)

    # Normalize and resize the secret image to match cover image size
    secret_img_resized = cv2.resize(secret_img, cover_img.shape[::-1])  # Resize secret image to cover size
    secret_img_normalized = cv2.normalize(secret_img_resized, None, 0, 255, cv2.NORM_MINMAX)  # Normalize

    # Step 2: DWT decomposition of secret image
    _, secret_LH, _, _ = dwt_decompose(secret_img_normalized)

    # Step 3: Embed the secret's LH subband into cover's LH subband using alpha blending
    embedded_LH = cover_LH + alpha * secret_LH

    # Step 4: Reconstruct the image using inverse DWT, but with the modified LH subband
    embedded_img = dwt_reconstruct(cover_LL, embedded_LH, cover_HL, cover_HH)

    return embedded_img


def DWT(cover_img, secret_img):
    I = [0, 40, 80, 90, 100]
    Stego_Img = []
    Reconstruct_Img = []
    for m in range(len(I)):
        Img = cover_img[I[m]]
        # Normalize cover image to ensure it's clear and bright
        cover_img = np.clip(cover_img, 0, 255).astype(np.uint8)  # Ensure that pixel values are in the valid range

        # Embed the secret image into the cover image
        embedded_img = embed_images(np.resize(Img, (512, 512)), secret_img, alpha=0.1)
        Stego_Img.append((embedded_img))
        Reconstruct_Img.append(Img)

    return np.asarray(Stego_Img), np.asarray(Reconstruct_Img)
