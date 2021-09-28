import torch
import torch.optim as optim
# epsilon = 2./255
# 
# delta = torch.zeros_like(pig_tensor, requires_grad=True)
# opt = optim.SGD([delta], lr=1e-1) # here we puted delta instead of our model
# 
# for t in range(30):
    # pred = model(norm(pig_tensor + delta)) # we try to predit our image w/ the attack mask
    # loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([341])) # pig
    # if t % 5 == 0:
        # print(t, loss.item())
    # 
    # opt.zero_grad()
    # loss.backward()
    # opt.step()
    # delta.data.clamp_(-epsilon, epsilon)
    # 
# print("True class probability:", nn.Softmax(dim=1)(pred)[0,341].item())


def _create_empty_mask(tensor: torch.tensor) -> torch.tensor:
    """
    Create a zero_like tensor based on the provided tensor
    """

    return torch.zeros_like(tensor, requires_grad=True)

def create_mask(model: torch.nn.Module, image: torch.tensor, image_class: torch.tensor) -> torch.tensor:
    epsilon = 2./255

    mask = _create_empty_mask(image)

    opt = optim.SGD([mask], lr=1e-1)
    EPOCH = 100

    for t in range(EPOCH):

        pred = model(image + mask)
        loss = - torch.nn.CrossEntropyLoss()(pred, image_class) # lower is better (including negative numbers)

        if t % (EPOCH / 10) == 0:
            print(t, loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        mask.data.clamp_(-epsilon, epsilon)

    print("final loss:", loss.item())

    return mask


def create_targeted_mask(model: torch.nn.Module, image: torch.tensor, image_class: torch.tensor, target_class: torch.tensor) -> torch.tensor:
    epsilon = 2./255

    mask = _create_empty_mask(image)

    opt = optim.SGD([mask], lr=1e-4)
    EPOCH = 1_200

    losses = []
    # opt = optim.Adam([mask], lr=1e-5)

    for t in range(EPOCH):
        opt.zero_grad()

        pred = model(image + mask)
        loss = (
            - torch.nn.CrossEntropyLoss()(pred, image_class)
            + torch.nn.CrossEntropyLoss()(pred, target_class)
        ) # lower is better (including negative numbers)

        losses.append(loss.item())

        if t % (EPOCH / 10) == 0:
            print(t, loss.item())
        
        loss.backward()

        opt.step()

        mask.data.clamp_(-epsilon, epsilon)

    print("final loss:", loss.item())

    import matplotlib.pyplot as plt
    plt.plot(range(EPOCH), losses)
    plt.show()

    return mask
