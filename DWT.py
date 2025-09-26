import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


# def load_image(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (512, 512))  # Resize to consistent size
#     return img


def dwt_decompose(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH


def dwt_reconstruct(LL, LH, HL, HH):
    coeffs = (LL, (LH, HL, HH))
    img_reconstructed = pywt.idwt2(coeffs, 'haar')
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)


def embed_images(cover_img, secret_img, alpha=0.1):
    # Step 1: DWT decomposition
    cover_LL, cover_LH, cover_HL, cover_HH = dwt_decompose(cover_img)
    secret_LL, _, _, _ = dwt_decompose(secret_img)

    # Step 2: Embed secret_LL into cover_LL using alpha blending
    embedded_LL = cover_LL + alpha * secret_LL

    # Step 3: Reconstruct the embedded image using inverse DWT
    embedded_img = dwt_reconstruct(embedded_LL, cover_LH, cover_HL, cover_HH)

    return embedded_img


def extract_secret(embedded_img, cover_img, alpha=0.1):
    # Step 1: DWT decomposition of embedded and cover images
    embedded_LL, _, _, _ = dwt_decompose(embedded_img)
    cover_LL, _, _, _ = dwt_decompose(cover_img)

    # Step 2: Extract secret image LL component using reverse operation
    extracted_LL = (embedded_LL - cover_LL) / alpha

    # Step 3: Reconstruct the secret image using inverse DWT
    extracted_img = dwt_reconstruct(extracted_LL, np.zeros_like(embedded_LL),
                                    np.zeros_like(embedded_LL), np.zeros_like(embedded_LL))

    return extracted_img


# Load cover and secret images
cover_img = np.load('Image.npy', allow_pickle=True)[0]
secret_img = np.load('Image.npy', allow_pickle=True)[0]

# Embed secret into cover
embedded_img = embed_images(np.resize(cover_img, (512, 512)), np.resize(secret_img, (512, 512)))

# Extract secret from embedded image
extracted_img = extract_secret(embedded_img, cover_img)

# Display the images
plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1)
plt.title('Cover Image')
plt.imshow(cover_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Secret Image')
plt.imshow(secret_img)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Embedded Image')
plt.imshow(embedded_img)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Extracted Image')
plt.imshow(extracted_img)
plt.axis('off')

plt.show()
