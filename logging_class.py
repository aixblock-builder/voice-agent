import binascii
import json
import os
import queue
import sys
import threading
import time

import requests

CENTRIFUGE_URL = "wss://rt.aixblock.io/centrifugo/connection/websocket"
CENTRIFUGE_SECRET = os.environ.get(
    "CENTRIFUGE_SECRET", "d0a70289-9806-41f6-be6d-f4de5fe298fb"
)
CENTRIFUGE_TOPIC_PREFIX = os.environ.get("CENTRIFUGE_TOPIC_PREFIX", "")
HOST_NAME = os.getenv("HOST_NAME", "https://app.aixblock.io")
TOKEN = os.getenv("TOKEN", "3cf7af3b5e87cd8674c548689ffa53561a2a8388")

HOST_NAME = "http://127.0.0.1:8080"
TOKEN = "f81b54e7b72cb6dc7e0c4367de3b8d8a59b6a093"

if CENTRIFUGE_TOPIC_PREFIX == "":
    if HOST_NAME == "":
        CENTRIFUGE_TOPIC_PREFIX = "prefix/"
    else:
        hostname_hash = str(hex(binascii.crc32(HOST_NAME.encode("utf-8"))))
        CENTRIFUGE_TOPIC_PREFIX = (
            hostname_hash[0:16] if len(hostname_hash) > 16 else hostname_hash
        ) + "/"


CENTRIFUGE_API = os.environ.get(
    "CENTRIFUGE_API", "https://rt.aixblock.io/centrifugo/api"
)
CENTRIFUGE_API_KEY = os.environ.get(
    "CENTRIFUGE_API_KEY", "ee5f81f5-0f68-48c7-a8e3-d790d92e0fd4"
)


def publish_message(channel, data, prefix=True, **kwargs):
    def p():
        topic = channel
        if prefix:
            topic = CENTRIFUGE_TOPIC_PREFIX + channel
        r = requests.post(
            CENTRIFUGE_API + "/publish",
            headers={"X-API-Key": CENTRIFUGE_API_KEY},
            data=json.dumps({"channel": topic, "data": data}),
        )

    t = threading.Thread(target=p, args=list())
    t.start()
    return t


class StreamLogger:
    def __init__(self, original_stream, log_queue):
        self.original_stream = original_stream  # Luồng gốc (stdout hoặc stderr)
        self.log_queue = log_queue  # Hàng đợi để chuyển log tới luồng khác

    def write(self, message):
        if message.strip():  # Chỉ ghi log nếu không phải dòng trống
            self.original_stream.write(message)  # Ghi ra terminal
            self.original_stream.flush()  # Đảm bảo được ghi ngay
            self.log_queue.put(message)  # Đưa log vào hàng đợi

    def flush(self):
        self.original_stream.flush()  # Gọi flush của luồng gốc

    def isatty(self):
        return self.original_stream.isatty()  # Giữ nguyên hành vi của isatty

    def fileno(self):
        return (
            self.original_stream.fileno()
        )  # Đảm bảo các thư viện khác vẫn sử dụng được


def log_worker(log_queue, channel):
    """Luồng chạy nền để xử lý log."""
    with open("real_time_logs.log", "w") as log_file:
        while True:
            try:
                message = log_queue.get()
                if message is None:
                    break  # Kết thúc luồng nếu nhận được tín hiệu dừng
                log_file.write(message)  # Ghi log vào file
                log_file.flush()  # Đảm bảo dữ liệu được ghi ngay lập tức
                publish_message(channel=channel, data={"log": message}, prefix=False)
            except Exception as e:
                print(f"Logging error: {e}", file=sys.__stderr__)


def start_queue(channel):
    # Tạo hàng đợi và luồng
    log_queue = queue.Queue()
    logging_thread = threading.Thread(target=log_worker, args=(log_queue, channel))
    logging_thread.daemon = True  # Đảm bảo luồng dừng khi chương trình kết thúc
    logging_thread.start()
    return log_queue, logging_thread


def write_log(log_queue):
    # Chuyển hướng sys.stdout và sys.stderr
    sys.stdout = StreamLogger(sys.stdout, log_queue)
    sys.stderr = StreamLogger(sys.stderr, log_queue)


# Code chính
# try:
#     print("Starting training...")  # Sẽ được ghi ra cả terminal và file
#     time.sleep(2)  # Mô phỏng training
#     print("Training is in progress...")
#     time.sleep(2)
#     print("Training completed!")
# except Exception as e:
#     print(f"An error occurred: {e}")


def stop_log(log_queue, logging_thread):
    # Dừng luồng log
    log_queue.put(None)
    logging_thread.join()

    # Khôi phục sys.stdout và sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# def reset_log():
#     # Khôi phục sys.stdout và sys.stderr
#     sys.stdout = sys.__stdout__
#     sys.stderr = sys.__stderr__

# print("Logging finished. Log saved to 'real_time_logs.log'")
