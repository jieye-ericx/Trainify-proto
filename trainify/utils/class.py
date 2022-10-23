def call_class_func(object, func_name):
    assert isinstance(func_name, str), 'func_name must be a string'
    if func_name not in object.__dict__:
        func = getattr(object, func_name, None)
        if func is not None:
            func()
            return
    print('object dont have function ' + func_name)
