import math
import re


class Test(object):
    def __init__(self, name):
        self.name = name
        self.xnames = ['thdot', 'cos', 'sin']
        self.x = [2, 2, 2]
        self.d = ["x[1]**4+2+x[2]+sin(x[2])", "x[0]+x[2]**3+x[1]", "x[1]*x[0]"]
        self.create_verify_func()

    def create_verify_func(self, Env):
        labels = [
            ['sin', 'math.sin'],
            ['cos', 'math.cos'],
            ['tan', 'math.tan'],
            ['tanh', 'math.tanh']
        ]

        def from_function(str, max):
            def cal(self, x):
                if max:
                    return eval(str)
                else:
                    return -eval(str)

            return cal

        for i, str in enumerate(self.d):
            for a in labels:
                reg = re.compile(re.escape(a[0]), re.IGNORECASE)
                self.d[i] = reg.sub(a[1], self.d[i])

            setattr(Env, self.xnames[i] + '_maximum',
                    from_function(self.d[i], True))
            setattr(Env, self.xnames[i] + '_minimum',
                    from_function(self.d[i], False))


a = Test("123")
print(a.cos_minimum(a.x), a.cos_maximum(a.x))
print(Test.__dict__)
