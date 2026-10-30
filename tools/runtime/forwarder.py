#!/usr/bin/env python3
"""Forward an agent's loopback HTTP proxy to the sidecar's shared Unix socket."""
import argparse
import select
import socket
import threading

def relay(client, path):
    upstream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        upstream.connect(path)
        peers={client:upstream,upstream:client}
        while True:
            readable,_,_=select.select(list(peers),[],[],60)
            if not readable:return
            for source in readable:
                data=source.recv(65536)
                if not data:return
                peers[source].sendall(data)
    finally:client.close();upstream.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--socket',default='/run/abench-egress/proxy.sock');ap.add_argument('--port',type=int,default=3128);a=ap.parse_args()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',a.port));listener.listen()
        while True:
            client,_=listener.accept();threading.Thread(target=relay,args=(client,a.socket),daemon=True).start()
if __name__=='__main__':main()
