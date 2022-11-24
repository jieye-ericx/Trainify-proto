import requests
import json

_APPKEY = "BC-c745270abeb04089a10632bbb6bec81b"
_GOEASY_URL = "http://rest-hangzhou.goeasy.io/publish"


def send_to_backend(channel, content):
    data = {"appkey": _APPKEY, "channel": channel, "content": json.dumps(content)}
    # print(data)
    return requests.post(_GOEASY_URL, data=data)
