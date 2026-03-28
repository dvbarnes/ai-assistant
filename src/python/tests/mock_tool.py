import dspy


class MockTool(dspy.Tool):
    def __init__(self,func, ret_value = None):
        super().__init__(func=func)
        self.__ret_value = ret_value
    def __call__(self, **kwargs):
        return self.__ret_value
