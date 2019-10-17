import torch
import torchvision
from PIL import Image
from torchvision import transforms

from cifar10_CNN import Net, prepare_dataset, imshow


def acc_score(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


def class_acc(classes, net, testloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]} %')


def main():
    classes = ('cat', 'car', 'plane', 'bird',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform2 = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # trainloader, testloader = prepare_dataset(transform)

    net = Net()
    net.load_state_dict(torch.load('./cifar_net.pth'))

    # 全体の正答率
    # acc_score(net, testloader)
    # クラス毎の正答率
    # class_acc(classes, net, testloader)

    # ブーンの推測
    boon_img = Image.open('./data/cifar-10-batches-py/boon.png')
    # boon_img.convert('RGB')
    boon_transformed = transform2(boon_img)
    boonloader = torch.utils.data.DataLoader(boon_transformed.unsqueeze(0))
    # for i in boonloader:
    #     print(i.shape)
    # image = iter(boon_transformed)
    # print(dataiter)
    # imshow(torchvision.utils.make_grid(boon_transformed))

    with torch.no_grad():
        for data in boonloader:
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            print(predicted.item())


if __name__ == '__main__':
    main()
