import torchvision
import torchvision.transforms as transforms
import torch

def load_data():
    #đây là phần xử lí ảnh, ảnh được đưa vào và sau đó xử lý
    #Mỗi ảnh sẽ có [H,W,C], mỗi pixel xe có 3 thông số màu thì totensor sẽ chia hết cho 255 để đưa về [0,1]
    #Normalize sẽ nhân 2 đâu vào là mean và std sau đó lấy từng sô ở trên (x-mean)/std cuối cùng là trở thành [-1,1] để dễ xử lí hơn.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    # cái này sẽ lấy dữ liệu ảnh trong thư mục root nếu chưa có data trong thư mục này thì sẽ toạn thư mục và tải data về trên internet theo thư viện
    # train = true là lấy dữ liệu theo train nó sẽ lấy khoảng 5000 ảnh
    # transform là nó sẽ gắn thêm cách xử lí nảh vào khi nào sử dụng đến ảnh đó thì nó sẽ xử lí sau.
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    # ở đây thì nó sẽ lấy các cái ảnh theo trainset vào chia theo batch mỗi batch là 64 ảnh , mà mỗi batch sẽ được xáo trộn shuffle
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)

    return trainloader, testloader