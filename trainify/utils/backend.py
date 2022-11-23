import requests
import json

_APPKEY = "BC-2ecd0636b9944bb6899144921a12af70"
_GOEASY_URL = "http://rest-hangzhou.goeasy.io/publish"


def send_to_backend(channel, content):
    data = {"appkey": _APPKEY, "channel": channel, "content": json.dumps(content)}
    # print(data)
    return requests.post(_GOEASY_URL, data=data)
