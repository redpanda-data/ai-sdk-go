"""Auth proxy for Vertex AI Conformance.

Usage: proxy.py BIND PORT UPSTREAM_HOST PATH_PREFIX
The access token is the first line on stdin. The proxy never refreshes it.
"""
import http.client
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIND, PORT, UPSTREAM, PREFIX = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
TOKEN = sys.stdin.readline().strip()
if not TOKEN:
    sys.exit("proxy: no token on stdin")

# Headers the proxy sets itself or that only make sense per connection. Without
# accept-encoding the upstream body comes back uncompressed, as the token
# redaction needs.
SKIP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
        "proxy-authorization", "proxy-authenticate", "host", "authorization",
        "content-length", "accept-encoding"}
count = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle_any(self):
        global count
        count += 1
        rid = count
        start = time.time()
        path = self.path
        if not path.startswith(PREFIX) or ".." in path or "%2e" in path.lower():
            self.send_error(403, "path not allowed")
            self.log(rid, path, "403 denied")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in SKIP}
        headers["Authorization"] = "Bearer " + TOKEN
        conn = http.client.HTTPSConnection(UPSTREAM, timeout=300)
        try:
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            self.send_error(502, type(exc).__name__)
            self.log(rid, path, "502 " + type(exc).__name__)
            return
        chunks = total = 0
        status = str(resp.status)
        try:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in SKIP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                data = resp.read1(8192)
                if not data:
                    break
                data = data.replace(TOKEN.encode(), b"[REDACTED]")
                chunks += 1
                total += len(data)
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception as exc:
            # A response cut halfway still gets its req=N line, marked aborted.
            status += " aborted " + type(exc).__name__
            self.close_connection = True
        finally:
            conn.close()
            self.log(rid, path, f"{status} chunks={chunks} bytes={total} t={time.time() - start:.2f}s")

    def log(self, rid, path, status):
        sys.stderr.write(f"req={rid} {self.command} {path.split('?')[0]} -> {status}\n")
        sys.stderr.flush()

    do_GET = do_POST = handle_any

    def log_message(self, *args):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


Server((BIND, PORT), Handler).serve_forever()
