import requests
import json

_APPKEY = "BC-0deba4e6a6374befa1ae802f62e3816b"
_GOEASY_URL = "http://rest-hangzhou.goeasy.io/publish"


def send_to_backend(channel, content):
    data = {"appkey": _APPKEY, "channel": channel, "content": json.dumps(content)}
    # print(data)
    return requests.post(_GOEASY_URL, data=data)
