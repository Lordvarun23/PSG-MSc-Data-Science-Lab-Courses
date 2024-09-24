import subprocess

# Step 2: Verify the digital signature using the public key
command = [
    "openssl", "dgst", "-sha256", "-verify", "mypublic.key", "-signature", "signature.bin", "message.txt"
]

try:
    # Run the OpenSSL command to verify the signature
    subprocess.run(command, check=True)
    print("Signature verification successful.")
except subprocess.CalledProcessError as e:
    print(f"Signature verification failed: {e}")
