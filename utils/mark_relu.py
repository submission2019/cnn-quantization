from torchvision.models.resnet import Bottleneck, BasicBlock
from torch.nn.parallel.data_parallel import DataParallel

def mark_bottlenetck_before_relu(model):
    for m in model.children():
        if isinstance(m, Bottleneck):
            m.conv1.before_relu = True
            m.bn1.before_relu = True
            m.conv2.before_relu = True
            m.bn2.before_relu = True
        else:
            mark_bottlenetck_before_relu(m)

def mark_basicblock_before_relu(model):
    for m in model.children():
        if isinstance(m, BasicBlock):
            m.conv1.before_relu = True
            m.bn1.before_relu = True
        else:
            mark_basicblock_before_relu(m)

def resnet_mark_before_relu(model):
    if isinstance(model, DataParallel):
        model.module.conv1.before_relu = True
    else:
        model.conv1.before_relu = True

    mark_bottlenetck_before_relu(model)
    mark_basicblock_before_relu(model)
