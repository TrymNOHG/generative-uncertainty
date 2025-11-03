import torch
from tqdm import trange
from utils.training_params import TrainingParameters
from models.base import BaseModel

import matplotlib.pyplot as plt

def train_model(model: BaseModel, trainloader: torch.utils.data.DataLoader, train_parameters: TrainingParameters, device: torch.device, save: bool = True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_parameters.learning_rate)
    loss_legend = model.loss_legend()
    total_losses = [[] for _ in range(len(loss_legend))]

    for epoch in trange(train_parameters.num_epochs):
        for x, y in trainloader:
            x = x.to(device)
            losses = model.loss(x, y)
            total_loss = losses[0]
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for i, loss in enumerate(losses):
                total_losses[i].append(loss.item())

    for loss in total_losses:
        plt.plot(loss)
    plt.legend(loss_legend)
    plt.show()

    if save:
        torch.save(model.state_dict(), f"./pretrained_models/{model.short_name()}.pth")

    