#!/usr/bin/env python

from huggingface_hub import configure_http_backend
import requests

def proxy(url="socks5h://127.0.0.1:1080"):
    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies.update({
            "http": url,
            "https": url
        })
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)