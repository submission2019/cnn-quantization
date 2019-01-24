from torch.nn.parallel.data_parallel import DataParallel


def module_type_to_string(m):
    return (str(type(m)).replace('>', '').replace('\'', '').split('.')[-1]).replace('WithId', '')


def set_node_name_recurcive(parent, parent_name, ldict=None):
    has_children = False
    for m in parent.named_children():
        has_children = True
        t = module_type_to_string(m[1])
        m_name = parent_name + '/' + t + '[' + m[0] + ']'
        set_node_name_recurcive(m[1], m_name, ldict=ldict)

    if not has_children:
        parent.internal_name = parent_name
        if ldict is not None:
            ldict[parent_name] = parent
        # print(parent_name)


def set_node_names(model, format='tensorboard', create_ldict=None):
    # Currently only tensorboard format supported
    ldict = {} if create_ldict else None
    set_node_name_recurcive(model, module_type_to_string(model), ldict=ldict)
    if create_ldict:
        return ldict
