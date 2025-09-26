import numpy as np
from PIL import Image


# Function to apply Arnold's Cat Map (encryption)
def arnold_encryption(img, iterations):
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    for _ in range(iterations):
        new_img_array = np.zeros_like(img_array)

        for x in range(width):
            for y in range(height):
                # Apply Arnold's cat map transformation
                new_x = (2 * x + y) % width
                new_y = (x + y) % height
                new_img_array[new_y, new_x] = img_array[y, x]

        img_array = new_img_array

    return img_array


# Function to apply the inverse Arnold's Cat Map (decryption)
def arnold_decryption(img, iterations):
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    for _ in range(iterations):
        new_img_array = np.zeros_like(img_array)

        for x in range(width):
            for y in range(height):
                # Apply inverse Arnold's cat map transformation
                new_x = (2 * x - y) % width
                new_y = (x - y) % height
                new_img_array[new_y, new_x] = img_array[y, x]

        img_array = new_img_array

    return img_array


# Main function to demonstrate the encryption and decryption
def process_image(image_path, iterations=5):

    # Encrypt image using Arnold's Cat Map
    encrypted_img_array = arnold_encryption(image_path, 5)
    # save_image(encrypted_img_array, 'encrypted_image.png')
    # print("Encrypted image saved as 'encrypted_image.png'.")

    # Decrypt image using inverse Arnold's Cat Map
    decrypted_img_array = arnold_decryption(encrypted_img_array, 5)
    # save_image(decrypted_img_array, 'decrypted_image.png')
    print("Decrypted image saved as 'decrypted_image.png'.")

