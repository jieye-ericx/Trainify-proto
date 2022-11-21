import logging
import logging.config
import requests

APPKEY = "BC-2ecd0636b9944bb6899144921a12af70"
GOEASY_URL = "http://rest-hangzhou.goeasy.io/publish"


class BackendHandler(logging.Handler, object):
    """
    后端实时输出handler
    """

    def __init__(self, channel, other_attr=None):
        logging.Handler.__init__(self)
        self.channel = channel
        # print('初始化自定义日志处理器：', channel)
        # print('其它属性值：', other_attr)

    def emit(self, record):
        """
        emit函数为自定义handler类时必重写的函数，这里可以根据需要对日志消息做一些处理，比如发送日志到服务器
        发出记录(Emit a record)
        """
        try:
            msg = self.format(record)
            content = {
                "type": "msg",
                "data": {
                    "content": msg
                }
            }
            data = {"appkey": APPKEY, "channel": self.channel, "content": content}
            res = requests.post(GOEASY_URL, data=data)
        except Exception:
            self.handleError(record)
