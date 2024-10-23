#!/usr/bin/env python

from huggingface_hub import configure_http_backend
import requests

# Create a factory function that returns a Session with configured proxies
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.proxies.update({
        "http": "socks5h://127.0.0.1:1080",
        "https": "socks5h://127.0.0.1:1080"
    })
    return session

# Set it as the default session factory
configure_http_backend(backend_factory=backend_factory)