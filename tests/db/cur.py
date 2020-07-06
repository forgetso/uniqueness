class TestCur:

    def __init__(self, data, descriptors):
        assert (isinstance(descriptors, list))
        self.description = descriptors
        self.data = list(data)


    def __iter__(self):
        return self.data.__iter__()
