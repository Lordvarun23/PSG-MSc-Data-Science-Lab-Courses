import subprocess

# OpenSSL command to encrypt the file

# openssl enc -aes-256-cbc -md sha512 -pbkdf2 -iter 1000 -salt -in message.txt -out encrypt.enc
command = [
    "openssl", "enc", "-aes-256-cbc", "-md", "sha512", "-pbkdf2", "-iter", "1000", "-salt",
    "-in", "message.txt", "-out", "encrypt_1.enc"
]

try:
    # Run the OpenSSL command
    subprocess.run(command, check=True)
    print("File encrypted and saved to 'encrypt.enc'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during encryption: {e}")
