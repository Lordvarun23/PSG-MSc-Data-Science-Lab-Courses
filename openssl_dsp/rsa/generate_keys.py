import subprocess

# Command to generate a private key with AES-256-CBC encryption
# openssl genrsa -aes-256-cbc -out myprivate.key
generate_private_key = [
    "openssl", "genrsa", "-aes-256-cbc", "-out", "myprivate.key"
]

# Command to generate a public key from the private key
# openssl rsa -in myprivate.key -pubout > mypublic.keys
generate_public_key = [
    "openssl", "rsa", "-in", "myprivate.key", "-pubout", "-out", "mypublic.key"
]

try:
    # Run the command to generate the private key
    subprocess.run(generate_private_key, check=True)
    print("Private key saved as 'myprivate.key'.")
    
    # Run the command to generate the public key
    subprocess.run(generate_public_key, check=True)
    print("Public key saved as 'mypublic.key'.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred during key generation: {e}")
