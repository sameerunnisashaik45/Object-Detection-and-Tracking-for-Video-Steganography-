import numpy as np
from Crypto.Util import number
import time
import tracemalloc
import sys


def generate_elgamal_keypair(bits=2048):
    # Generate a large prime p
    p = number.getPrime(bits)

    # Select a primitive root g
    g = number.getRandomRange(2, p - 1)  # Example, you might need to find a real primitive root

    # Choose private key x
    x = number.getRandomRange(1, p - 2)

    # Compute the public key y = g^x mod p
    y = pow(g, x, p)

    private_key = (p, g, x)  # Private key is (p, g, x)
    public_key = (p, g, y)  # Public key is (p, g, y)

    return public_key, private_key


def elgamal_encrypt(public_key, plaintext):
    p, g, y = public_key
    m = int.from_bytes(plaintext.encode(), byteorder='big')  # Message as an integer

    # Select a random number k
    k = number.getRandomRange(1, p - 1)

    # Compute c1 = g^k mod p
    c1 = pow(g, k, p)

    # Compute c2 = m * y^k mod p
    c2 = (m * pow(y, k, p)) % p

    # Return the ciphertext (c1, c2)
    return c1, c2


def elgamal_decrypt(private_key, ciphertext):
    p, g, x = private_key
    c1, c2 = ciphertext

    # Compute the inverse of c1^x mod p
    s = pow(c1, x, p)
    s_inv = pow(s, p - 2, p)  # Using Fermat's little theorem to find the modular inverse

    # Decrypt the message
    m = (c2 * s_inv) % p

    # Convert the decrypted message back to bytes and then decode
    decrypted_message = m.to_bytes((m.bit_length() + 7) // 8, byteorder='big').decode()
    return decrypted_message


def get_memory_size(obj):
    return sys.getsizeof(obj)


def ElGamal(data):
    plaintext = str(data)

    # Generate ElGamal keypair
    public_key, private_key = generate_elgamal_keypair(bits=2048)

    tracemalloc.start()
    ct = time.time()

    # Encrypt the message
    ciphertext = elgamal_encrypt(public_key, plaintext)
    ENC_time = time.time() - ct

    # Decrypt the message
    decrypted_plaintext = elgamal_decrypt(private_key, ciphertext)

    mem_size = get_memory_size(plaintext)
    DEC_Time = time.time() - ENC_time
    Compt_Time = ENC_time + DEC_Time

    return [ENC_time, DEC_Time, mem_size, Compt_Time]

