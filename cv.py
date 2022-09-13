import numpy as np
import torch,tqdm,os,sys,random,time
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
from torchvision import transforms,datasets
from models import *
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
env_seed = int(sys.argv[1])
random.seed(env_seed)
np.random.seed(env_seed)
torch.manual_seed(env_seed)
torch.cuda.manual_seed_all(env_seed)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print('==> Preparing data..')
datasetname, netname, batch_size = str(sys.argv[2]),str(sys.argv[3]),int(sys.argv[4])
lr, momentum, gamma, weight_decay = round(float(sys.argv[5]),2),round(float(sys.argv[6]),2),round(float(sys.argv[7]),2),5e-4#1e-6
step_size, max_epoch= int(sys.argv[8]),int(sys.argv[9])
def updatelr(epoch):
    scheduler.step()
    return
    eta_min_ratio = 0.01
    lr_crt = lr*((1-eta_min_ratio)*(1-epoch/max_epoch)+eta_min_ratio)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#netname, batch_size, lr, momentum, step_size, gamma = 'vgg11', 500, 0.1, 0.9, 150, 0.1
print('==> Building model..',netname)
if netname == 'vgg11': net = VGG('VGG11')
if netname == 'vgg13': net = VGG('VGG13')
if netname == 'vgg16': net = VGG('VGG16')
if netname == 'vgg19': net = VGG('VGG19')
if netname == 'res18': net = ResNet18()
if netname == 'res34': net = ResNet34()
if netname == 'res50': net = ResNet50()
if netname == 'res101':net = ResNet101()
if netname == 'res152':net = ResNet152()
if netname == 'pre': net = PreActResNet18()
if netname == 'gln': net = GoogLeNet()
if netname == 'den': net = DenseNet121()
if netname == 'rex': net = ResNeXt29_2x64d()
if netname == 'mob': net = MobileNet()
if netname == 'mo2': net = MobileNetV2()
if netname == 'dpn': net = DPN92()
if netname == 'shf': net = ShuffleNetG2()
if netname == 'sen': net = SENet18()
if netname == 'sh2': net = ShuffleNetV2(1)
if netname == 'eff': net = EfficientNetB0()
if netname == 'reg': net = RegNetX_200MF()
if netname == 'sim': net = SimpleDLA()
if netname == 'lenet': net = LeNet()
if netname == 'rnn': net = RNN()
net = net.cuda()#net = torch.nn.DataParallel(net)
if datasetname=='cinic':
    cinic_mean, cinic_std = [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()
                                    ,transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])
    trainset0 = datasets.ImageFolder(root='./myenv/data/cinic/train', transform=train_transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validateset = datasets.ImageFolder(root='./myenv/data/cinic/valid', transform=transform)
    #validateloader = torch.utils.data.DataLoader(validateset, batch_size=batch_size, shuffle=True, num_workers=4)
    trainset = torch.utils.data.ConcatDataset([trainset0,validateset])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = datasets.ImageFolder(root='./myenv/data/cinic/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    #classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if datasetname=='cifar':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()
                                    ,transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                                    #,transforms.ToTensor(),transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])])
    trainset = datasets.CIFAR10(root='./myenv/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = datasets.CIFAR10(root='./myenv/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if datasetname=='mnist':
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    trainset = datasets.MNIST(root='./myenv/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = datasets.MNIST(root='./myenv/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    #classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
print('trainset length:',len(trainset),'testset length:',len(testset))
fulllength = len(trainset)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)#, weight_decay=5e-4*batch_size)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
label = datasetname+'-'+netname+'-'+str(lr)+','+str(momentum)+'-'+str(step_size)+','+str(gamma)+'-'+str(batch_size)
foldername = './results/'
os.makedirs(foldername,exist_ok=True)
ftrain,ftest=open(foldername+'train','a'),open(foldername+'test','a')
print('',file=ftrain,flush=True)
print(label,end='|',file=ftrain,flush=True)
print('',file=ftest,flush=True)
print(label,end='|',file=ftest,flush=True)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss,correct,total = 0,0,0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #inputs = np.tile(inputs,3)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(epoch*fulllength+batch_idx*batch_size,',',round(100.*correct/total,3),end='|',file=ftrain,flush=True)
def test(epoch):
    net.eval()
    test_loss,correct,total = 0,0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print((epoch+1)*fulllength,',',round(100.*correct/total,3),end='|',file=ftest,flush=True)
for epoch in tqdm.tqdm(range(max_epoch)):
    train(epoch)
    test(epoch)
    updatelr(epoch)
