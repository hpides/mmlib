import torch


def imagenet_input(batch_size=10):
    batch = []
    for i in range(batch_size):
        batch.append(torch.rand(3, 300, 400))
    return torch.stack(batch)


def blackbox_equals(m1, m2, produce_input):
    inp = produce_input()

    m1.eval()
    m2.eval()

    out1 = m1(inp)
    out2 = m2(inp)

    return torch.equal(out1, out2)


def whitebox_equals(m1, m2):
    state1 = m1.state_dict()
    state2 = m2.state_dict()

    return state_dict_equals(state1, state2)


def state_dict_equals(d1, d2):
    for item1, item2 in zip(d1.items(), d2.items()):
        layer_name1, weight_tensor1 = item1
        layer_name2, weight_tensor2 = item2
        if not layer_name1 == layer_name2 or not torch.equal(weight_tensor1, weight_tensor2):
            return False

    return True


def equals(m1, m2, produce_input):
    # whitebox and blackbox check should be redundant,
    # but this way we have an extra safety net in case we forgot a special case
    return whitebox_equals(m1, m2) and blackbox_equals(m1, m2, produce_input)
