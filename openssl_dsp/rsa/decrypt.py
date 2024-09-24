import subprocess

# OpenSSL command to decrypt using the private key
# openssl pkeyutl -decrypt -in encrypt.enc -inkey myprivate.key -out decrypt.txt
command = [
    "openssl", "pkeyutl", "-decrypt", "-in", "encrypt.enc", "-inkey", "myprivate.key", "-out", "decrypt.txt"
]

try:
    # Run the OpenSSL command to decrypt the file
    subprocess.run(command, check=True)
    print("File decrypted and saved to 'decrypt.txt'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during decryption: {e}")
