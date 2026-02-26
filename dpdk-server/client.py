import socket
import struct
import time
from aeon.datasets import load_arrow_head

SERVER_IP = "192.168.40.31"   # DPDK machine
SERVER_PORT = 9000

CLIENT_IP = "192.168.40.30"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 9001))  # local port for receiving reply

value = 108.108

X, y = load_arrow_head(split="test")
print(f"Loaded ArrowHead dataset: X shape {X.shape}, y shape {y.shape}")

stream = X.flatten()
print(f"Stream length: {len(stream)}")
# stream = stream[:len(X)]
# print(f"Modified Stream length: {len(stream)}")
i = 0

start = time.time()
for i in range(len(stream)):  # Send first 100 values of the stream
    payload = struct.pack("f", stream[i])  # Send first value of the stream
    sock.sendto(payload, (SERVER_IP, SERVER_PORT))
    data, _ = sock.recvfrom(1024)
    reply = struct.unpack("f", data[:4])[0]
    #print("Sent:", stream[i], "Received:", reply)
end = time.time()
print(f"Sent {i+1} values in {end - start:.2f} seconds")