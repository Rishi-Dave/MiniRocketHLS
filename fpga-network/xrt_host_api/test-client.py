import socket
import time
import struct
import numpy as np

CLIENT_IP = "192.168.40.31"   # enp134s0f0
CLIENT_PORT = 5005

SERVER_IP = "192.168.40.29"   # peer on same subnet
SERVER_PORT = 6006


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind client socket to NIC's IP
sock.bind((CLIENT_IP, CLIENT_PORT))  # 0 = ephemeral port

for i in range(5):
    msg = 1
    payload = struct.pack('<I', msg)          # little-endian float32
    sock.sendto(payload, (SERVER_IP, SERVER_PORT))
    print("TX:", msg)
