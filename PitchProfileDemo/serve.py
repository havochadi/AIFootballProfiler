"""Serve the demo on http://localhost:8000.

Static files come from this folder. /footage streams the match video straight from the data
drive (with Range support so the player can seek); the video is never copied into this folder.

    python serve.py [port]
"""
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
VIDEO = re.search(r'"video":\{"path":(".*?")', open(os.path.join(HERE, 'data.js'), encoding='utf-8').read()).group(1)
VIDEO = json.loads(VIDEO)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def do_GET(self):
        if self.path.split('?')[0] == '/footage':
            return self.footage()
        return super().do_GET()

    def footage(self):
        if not os.path.exists(VIDEO):
            self.send_error(404, 'Footage not found (is the data drive connected?)')
            return
        size = os.path.getsize(VIDEO)
        start, end = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        if m:
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            elif m.group(2):
                start = max(size - int(m.group(2)), 0)
        self.send_response(206 if m else 200)
        self.send_header('Content-Type', 'video/x-matroska')
        self.send_header('Accept-Ranges', 'bytes')
        if m:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        try:
            with open(VIDEO, 'rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, fmt, *args):
        if '/footage' not in (args[0] if args else ''):
            super().log_message(fmt, *args)


if __name__ == '__main__':
    print(f'PitchProfile demo on http://localhost:{PORT}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
