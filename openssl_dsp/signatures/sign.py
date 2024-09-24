import subprocess

# Step 1: Generate a digital signature using the private key
command = [
    "openssl", "dgst", "-sha256", "-sign", "myprivate.key", "-out", "signature.bin", "message.txt"
]

try:
    # Run the OpenSSL command to generate the signature
    subprocess.run(command, check=True)
    print("Digital signature generated and saved as 'signature.bin'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during signature generation: {e}")
