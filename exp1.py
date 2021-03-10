import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

directory = os.path.dirname(os.path.realpath(__file__))


# ImageFolderWithPaths class is from: <https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d>
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

image_width = 320
image_height = 180

SCALE_FACTOR = 2

def loader_transform():
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((180 // SCALE_FACTOR, 320 // SCALE_FACTOR)), transforms.ToTensor()])

trainset = ImageFolderWithPaths(
    root=os.path.join(directory, "../experiment1/training"),
    transform=loader_transform()
)

testset = ImageFolderWithPaths(
    root=os.path.join(directory, "../experiment1/test"),
    transform=loader_transform()
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False
)

device = torch.device('cpu')

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()

        # Source image dimensions (grayscale): 320x180
        image_width = 320 // SCALE_FACTOR
        image_height = 180 // SCALE_FACTOR

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # The width and height of the input after the conv steps
        w = (((image_width - 4) // 2) - 4) // 2
        h = (((image_height - 4) // 2) - 4) // 2

        # now a few fully connected layers
        self.fc1 = nn.Linear(16 * w * h, 120)
        self.dropout1 = nn.Dropout(p=0.75)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def eval_testset(model):
    with torch.no_grad():
        printed_first_incorrect = False

        n_correct = 0
        n_samples = 0
        for images, labels, paths in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            n_samples += labels.size(0)
            n_correct += correct

            if not printed_first_incorrect and correct < labels.size(0):
                printed_first_incorrect = True
                print(predicted, labels)
                print(paths)

        acc = 100.0 * n_correct / n_samples
        print(f'Num correct: {n_correct}/{n_samples}')
        print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

        return acc


learning_rate = 0.01
momentum = 0

model = NNModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

num_epochs = 12

loss_line = ([], []) # x,y points that will be plotted
acc_line = ([], []) # x,y points that will be plotted

acc_line[0].append(0)
acc_line[1].append(eval_testset(model))

n_total_iterations = len(trainloader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels, paths) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % (n_total_iterations // 16) == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (n_total_iterations // 16)))

            loss_line[0].append(epoch + ((i+1) / n_total_iterations))
            loss_line[1].append(running_loss / (n_total_iterations // 16))

            running_loss = 0.0

    acc_line[0].append(epoch + 1)
    acc_line[1].append(eval_testset(model))

plt.clf()

# two-scale plot instructions: <https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html>

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('epoch')
ax1.set_ylim([-0.05, 1.05])
ax1.set_ylabel('training loss', color=color)
ax1.plot(loss_line[0], loss_line[1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('test data accuracy', color=color)  # we already handled the x-label with ax1
ax2.set_ylim([-5, 105])
ax2.plot(acc_line[0], acc_line[1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig("graph1.png")

print("Done")
