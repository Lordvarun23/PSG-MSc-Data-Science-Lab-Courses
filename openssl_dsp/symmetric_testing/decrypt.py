import subprocess

# OpenSSL command to decrypt the file

# openssl enc -aes-256-cbc -md sha512 -pbkdf2 -iter 1000 -salt -in encrypt.enc -out decrypt.txt -d
command = [
    "openssl", "enc", "-aes-256-cbc", "-md", "sha512", "-pbkdf2", "-iter", "1000", "-d",
    "-in", "encrypt_1.enc", "-out", "decrypt_1.txt"
]

try:
    # Run the OpenSSL command
    subprocess.run(command, check=True)
    print("File decrypted and saved to 'decrypt.txt'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during decryption: {e}")
