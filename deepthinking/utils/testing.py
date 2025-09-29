""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import einops
import torch
from icecream import ic
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(net, loaders, mode, iters, problem, device, plot):
    accs = []
    for loader in loaders:
        if mode == "default":
            accuracy = test_default(net, loader, iters, problem, device, plot)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device, plot)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    # if problem == "mazes":
    #     predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    # elif problem == "chess":
    #     outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
    #     top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
    #     top_2 = einops.repeat(top_2, "n -> n k", k=8)
    #     top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
    #     outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
    #     outputs[:, 0] = -float("Inf")
    #     predicted = outputs.argmax(1)

    return predicted

def make_gif(inputs, targets, all_outputs, i):
    if hasattr(inputs, 'cpu'):
        inputs = inputs.detach().cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.detach().cpu().numpy()
    if hasattr(all_outputs, 'cpu'):
        all_outputs = all_outputs.detach().cpu().numpy()
    output_dir = "animation_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    batch_size = inputs.shape[0]
    for batch_idx in range(batch_size):
        input_data = inputs[batch_idx]
        output_frames = all_outputs[batch_idx,:,1]
        target_data = targets[batch_idx].reshape(3, 3)
        fig, ax = plt.subplots()
        num_animation_frames = 1 + len(output_frames) + 1
        def update(frame_number):
            ax.clear()
            num_intermediate_frames = output_frames.shape[0]
            if frame_number == 0:
                ax.plot(np.arange(len(input_data)), input_data, color='blue')
                ax.set_title("Input")
            elif 1 <= frame_number <= num_intermediate_frames:
                image_index = frame_number - 1
                ax.imshow(output_frames[image_index], cmap='gray_r')
                ax.set_title(f"Intermediate Frame {image_index + 1}")
            else:
                ax.imshow(target_data, cmap='gray_r')
                ax.set_title("Real Answer", color='green')
            ax.set_xticks([])
            ax.set_yticks([])
        ani = FuncAnimation(fig, update, frames=num_animation_frames, interval=1000, repeat=True)
        unique_id = (i * batch_size) + batch_idx
        filename = os.path.join(output_dir, f"animation_sample_{unique_id}.gif")
        ani.save(filename, writer='pillow', fps=1)
        plt.close(fig)
    return

def test_default(net, testloader, iters, problem, device, plot):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(testloader, leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            all_outputs = net(inputs, iters_to_do=max_iters)
            if plot:
                make_gif(inputs,targets,all_outputs,i)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)
            i += 1

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc
