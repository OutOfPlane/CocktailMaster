from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json
import dzdScale as dzdScale

# Global variable to store the latest weight
dzd = dzdScale.Scale("COM4")
dzd.init()
# 1. BACKGROUND THREAD TO READ HARDWARE CONSTANTLY
def read_scale_hardware():
    global current_weight
    while True:
        dzd.read_weight()
        

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