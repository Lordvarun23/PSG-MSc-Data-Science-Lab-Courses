import subprocess

# OpenSSL command to encrypt using the public key
# openssl pkeyutl -encrypt -in message.txt -pubin -inkey mypublic.key -out encrypt.enc
command = [
    "openssl", "pkeyutl", "-encrypt", "-in", "message.txt", "-pubin", "-inkey", "mypublic.key", "-out", "encrypt.enc"
]

try:
    # Run the OpenSSL command to encrypt the file
    subprocess.run(command, check=True)
    print("File encrypted and saved to 'encrypt.enc'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during encryption: {e}")
