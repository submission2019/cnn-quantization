
class DummyQuantizer:
    def __call__(self, tensor, tag="", stat_id=None, override_att=None):
        return tensor
