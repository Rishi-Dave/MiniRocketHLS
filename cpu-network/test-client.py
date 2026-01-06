import socket
import time

CLIENT_IP = "192.168.40.31"   # enp134s0f0
SERVER_IP = "192.168.40.30"   # peer on same subnet
SERVER_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind client socket to NIC's IP
sock.bind((CLIENT_IP, 0))  # 0 = ephemeral port

for i in range(5):
    msg = f"packet {i}"
    sock.sendto(msg.encode(), (SERVER_IP, SERVER_PORT))
    print("TX:", msg)
    time.sleep(1)

