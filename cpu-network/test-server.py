import socket

SERVER_IP = "192.168.40.30"   # enp134s0f0
SERVER_PORT = 5005
BUFFER_SIZE = 2048

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind explicitly to the NIC's IP
sock.bind((SERVER_IP, SERVER_PORT))

print(f"UDP server listening on {SERVER_IP}:{SERVER_PORT}")

while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    print(f"RX {len(data)} bytes from {addr}: {data.decode(errors='ignore')}")

