from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json
import dzdScale as dzdScale

SCALE_PORT = "COM4"

def connect_scale():
    """Open and initialize the scale, retrying until it succeeds.

    On a fresh restart the serial port may still be held by the previous
    process for a moment, so we keep retrying instead of crashing.
    """
    while True:
        try:
            scale = dzdScale.Scale(SCALE_PORT)
            scale.init()
            print("Scale connected on", SCALE_PORT)
            return scale
        except Exception as e:
            print("Scale connect failed, retrying...", e)
            time.sleep(1)

# Global variable to store the latest weight
dzd = connect_scale()

# 1. BACKGROUND THREAD TO READ HARDWARE CONSTANTLY
def read_scale_hardware():
    global dzd
    while True:
        try:
            dzd.read_weight()
        except Exception:
            # Try to recover the connection without killing the thread.
            try:
                dzd.port.close()
            except Exception:
                pass
            time.sleep(0.5)
            dzd = connect_scale()

# Start the background thread immediately
hardware_thread = threading.Thread(target=read_scale_hardware, daemon=True)
hardware_thread.start()


# 2. ULTRA-LIGHTWEIGHT HTTP SERVER
class ScaleHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/weight':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Instantly return the global variable without waiting on hardware
            response = {"grams": dzd.value, "stable": dzd.stable}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    # Silences the default spammy terminal logging so your Pi stays fast
    def log_message(self, format, *args):
        return

def run():
    server_address = ('', 8003) # Run scale on port 8003
    httpd = HTTPServer(server_address, ScaleHTTPHandler)
    print("Scale service running on port 8003...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()