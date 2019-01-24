from torch.autograd import Function

# f is any callable object

# attacher to forward
class attach_to_forward_class(Function):
    @staticmethod
    def forward(ctx, tensor, f, tag):
        # print('forward')
        # we want that output will have different id from input
        ctx.tag = tag
        return 1*f(tensor, tag)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# attacher to backward
class attach_to_backward_class(Function):
    @staticmethod
    def forward(ctx, tensor, f, tag):
        ctx.f = f
        ctx.tag = tag
        return 1*tensor

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        f = ctx.f
        return f(grad_output, ctx.tag), None, None


# attacher to backward
class attach_to_forward_backward_class(Function):
    @staticmethod
    def forward(ctx, tensor, f, b, tag):
        # print('forward')
        ctx.f = f
        ctx.b = b
        ctx.tag = tag
        return f(tensor, tag)

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        return ctx.b(grad_output, ctx.tag), None, None, None


# attacher to forward and backward
def pytorch_attach(tensor, f=None, b=None, tag=''):
    if f is not None and b is not None:
        tensor = attach_to_forward_backward_class.apply(tensor, f, b, tag)
    elif f is not None:
        tensor = attach_to_forward_class.apply(tensor, f, tag)
    elif b is not None:
        tensor = attach_to_backward_class.apply(tensor, b, tag)
    return tensor


