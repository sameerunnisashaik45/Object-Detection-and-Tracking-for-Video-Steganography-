import random
from PIL import Image
from Crypto.Util import number
import numpy as np


# Function to generate ElGamal keys
def generate_keys(bit_length=256):
    p = number.getPrime(bit_length)
    g = random.randint(2, p - 1)

    # Private key (x)
    x = random.randint(2, p - 2)

    # Public key (h = g^x mod p)
    h = pow(g, x, p)

    return p, g, h, x


# encrypt
def encrypt(p, g, h, m):
    k = random.randint(2, p - 2)  # Random number for each encryption
    c1 = pow(g, k, p)  # Calculate c1 = g^k mod p
    c2 = (m * pow(h, k, p)) % p  # Calculate c2 = m * h^k mod p
    return c1, c2


# Function to decrypt a single ciphertext (c1, c2)
def decrypt(p, x, c1, c2):
    s = pow(c1, x, p)  # Calculate s = c1^x mod p
    s_inv = pow(s, p - 2, p)  # Modular inverse of s (s^(p-2) mod p)
    m = (c2 * s_inv) % p  # Decrypt the message m = c2 * s_inv mod p
    return m


# Function to convert image to grayscale pixel values
def image_to_pixels(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    pixels = np.array(img)
    return pixels


# Function to convert numerical pixel data back to an image
def pixels_to_image(pixels, output_path):
    img = Image.fromarray(pixels)
    img.save(output_path)


# Function to encrypt the image using ElGamal
def encrypt_image(image_path, p, g, h):
    # pixels = image_to_pixels(image_path)
    encrypted_pixels = []

    for row in image_path:
        encrypted_row = []
        for pixel in row:
            c1, c2 = encrypt(p, g, h, pixel)
            encrypted_row.append((c1, c2))
        encrypted_pixels.append(encrypted_row)

    return encrypted_pixels, image_path.shape


# Function to decrypt the encrypted image using ElGamal
def decrypt_image(encrypted_pixels, original_shape, p, x):
    decrypted_pixels = np.zeros(original_shape, dtype=int)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            c1, c2 = encrypted_pixels[i][j]
            decrypted_pixel = decrypt(p, x, c1, c2)
            decrypted_pixels[i][j] = decrypted_pixel

    return decrypted_pixels


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


def Elgamal_Arnold(image_path):
    # Generate ElGamal keys
    p, g, h, x = generate_keys(bit_length=256)

    # Encrypt the image
    encrypted_pixels, original_shape = encrypt_image(image_path, p, g, h)
    # Encrypt image using Arnold's Cat Map
    encrypted_img_array = arnold_encryption(encrypted_pixels, 5)

    # Decrypt image using inverse Arnold's Cat Map
    decrypted_img_array = arnold_decryption(encrypted_img_array, 5)
    # Decrypt the image
    decrypted_pixels = decrypt_image(decrypted_img_array, original_shape, p, x)

    return encrypted_img_array, decrypted_pixels

