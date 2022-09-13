import numpy as np
import random, gym, gym.spaces, json, easydict, time, cv2, scipy
from itertools import cycle
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from myenv import *
import torch,torchvision
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
#from scipy.signal import convolve2d
#from skimage.measure import block_reduce
class VISION(gym.Env):
    def __init__(self):
        super().__init__()
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = easydict.EasyDict(json.load(envinfo_file))
        args = envinfo_args_dict
        datasetname = args.env_type
        batch_size = args.env_num
        if datasetname=='cinic':
            data_mean, data_std = [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
            trainset0 = torchvision.datasets.ImageFolder(root='./myenv/data/cinic/train', transform=train_transform)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            validateset = torchvision.datasets.ImageFolder(root='./myenv/data/cinic/valid', transform=transform)
            #validateloader = torch.utils.data.DataLoader(validateset, batch_size=batch_size, shuffle=True, num_workers=4)
            trainset = torch.utils.data.ConcatDataset([trainset0,validateset])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.ImageFolder(root='./myenv/data/cinic/test', transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
            classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if datasetname=='cifar':
            data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
            trainset = torchvision.datasets.CIFAR10(root='./myenv/data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR10(root='./myenv/data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if datasetname=='mnist':
            data_mean, data_std = (0.5, ), (0.5, ) # (0.1307,), (0.3081,)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
            trainset = torchvision.datasets.MNIST(root='./myenv/data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            testset = torchvision.datasets.MNIST(root='./myenv/data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
            classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
        print('trainset',len(trainset),'testset',len(testset))
        self.trainset,self.trainloader,self.testset,self.testloader,self.classes=trainset,trainloader,testset,testloader,classes
        self.batch_size = batch_size
        self.record_step= 10
        foldername = args.exp_dir+'rewards/'
        os.makedirs(foldername,exist_ok=True)
        self.ftrain=open(foldername+'0_0','a')
        self.ftest =open(foldername+'0_1','a')
        self.g_step = 0
        self.testing = True
        self.testingrewards = []
        obs = self.reset()
        observation_shape = obs.shape[1:]
        action_shape = len(self.classes)
        #observation_shape = [7,1,1]#[5,3,1]#[7,7,1]#[28,28,1]
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=observation_shape,dtype=np.uint8)
        self.action_space      = gym.spaces.Discrete(action_shape)
        self.reward_range      = [0,1]
        self.attr = {}
    def reset(self):
        if self.testing:
            self.dataiter = iter(self.testloader)
        else:
            print(int(self.g_step),',',int(np.mean(self.testingrewards)*10000),end='|',file=self.ftest,flush=True)
            print(int(self.g_step),',',int(np.mean(self.testingrewards)*10000))
            self.testingrewards = []
            self.dataiter = iter(self.trainloader)
        self.images, self.labels = self.dataiter.next()
        images = self.image_wrapper(self.images)

        # print(images.shape)
        # #print(images[0])
        # print(np.min(images),np.max(images))
        # plt.figure()
        # plt.hist(images.flatten(),bins='auto')
        # plt.show()
        # for i in range(5):
        #     print(np.min(images[i]),np.max(images[i]))
        #     #unique,counts = np.unique(images,return_counts=True)
        #     #print(np.asarray((unique,counts)).T)
        #     #print(np.histogram(images, bins=[0, 1, 2, 3]))
        #     plt.figure()
        #     plt.hist(images[i].flatten(),bins='auto')
        #     plt.show()
        # exit()
        return images
    def step(self, action):
        #print('step:',self.g_step,self.testing)
        reward = np.zeros(self.labels.shape[0])
        for i in range(len(reward)):
            if action[i]==self.labels[i]: reward[i] = 1
        if self.testing:
            self.testingrewards.append(reward)
            info = {'labels':[]}
        else:
            if self.g_step//self.batch_size%self.record_step==0: # g_step increase batch_num each time
                print(int(self.g_step),',',int(np.mean(reward)*10000),end='|',file=self.ftrain,flush=True)
            self.g_step += self.batch_size
            info = {'labels':self.labels.numpy()}
        try:
            self.images, self.labels = self.dataiter.next()
            images = self.image_wrapper(self.images)
            #assert images.shape[0]== self.batch_size
        except:
            if self.testing: self.testing = False
            else:            self.testing = True
            images = self.reset()
        if images.shape[0]!= self.batch_size:
            if self.testing: self.testing = False
            else:            self.testing = True
            images = self.reset()
        done = np.array([False for i in range(self.batch_size)])
        return images, reward, done, info
    def image_wrapper(self, images):
        images = np.transpose(images.numpy(),(0,2,3,1)) # to batch width length channel
        if images.shape[-1]==1: images = np.tile(images,3)
        #images = (images/2+0.5)*255
        #images = images.astype(np.uint8)
        #print(images.shape)
        return images
        # customized input could add here
        images_new = []
        for i in range(images.shape[0]):
            image = images[i]
            image = torch.nn.functional.avg_pool2d(image,2)
            image = torch.nn.functional.avg_pool2d(image,2)
            images_new.append(image.numpy())
        images_new = np.array(images_new)
        images_new = torch.from_numpy(images_new)
        #print(images_new.shape)
        return images_new
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        #npimg = np.where(npimg>0.135, 0.9, 0.0)
        fig = plt.figure(figsize=(16,16),tight_layout=True)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        exit()
    def render(self, mode='rgb_array', close=False):
        return None
        edge_length = int(np.sqrt(len(self.images)-1))+1
        labels = torch.zeros([edge_length*edge_length],dtype=self.labels.dtype)
        labels[:len(self.labels)] = self.labels
        labels = labels.reshape(edge_length,edge_length)
        for i in range(edge_length):
            print(' '.join('%8s' % self.classes[labels[i][j]] for j in range(edge_length)))
        self.imshow(torchvision.utils.make_grid(self.images,nrow=edge_length))
        return None
    def close(self):
        print('',file=self.ftrain,flush=True)
        self.ftrain.close()
        print('',file=self.ftest,flush=True)
        self.ftest.close()
    def seed(self, seed=None):
        random.seed(seed)


"""
        image0 = [[1,1,1],
                  [1,0,1],
                  [1,0,1],
                  [1,0,1],
                  [1,1,1]]
        image1 = [[0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0]]
        image2 = [[1,1,1],
                  [0,0,1],
                  [1,1,1],
                  [1,0,0],
                  [1,1,1]]
        image3 = [[1,1,1],
                  [0,0,1],
                  [1,1,1],
                  [0,0,1],
                  [1,1,1]]
        image4 = [[1,0,1],
                  [1,0,1],
                  [1,1,1],
                  [0,0,1],
                  [0,0,1]]
        image5 = [[1,1,1],
                  [1,0,0],
                  [1,1,1],
                  [0,0,1],
                  [1,1,1]]
        image6 = [[1,1,1],
                  [1,0,0],
                  [1,1,1],
                  [1,0,1],
                  [1,1,1]]
        image7 = [[1,1,1],
                  [1,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1]]
        image8 = [[1,1,1],
                  [1,0,1],
                  [1,1,1],
                  [1,0,1],
                  [1,1,1]]
        image9 = [[1,1,1],
                  [1,0,1],
                  [1,1,1],
                  [0,0,1],
                  [1,1,1]]
        image0 = np.array(image0)[np.newaxis,:]
        image1 = np.array(image1)[np.newaxis,:]
        image2 = np.array(image2)[np.newaxis,:]
        image3 = np.array(image3)[np.newaxis,:]
        image4 = np.array(image4)[np.newaxis,:]
        image5 = np.array(image5)[np.newaxis,:]
        image6 = np.array(image6)[np.newaxis,:]
        image7 = np.array(image7)[np.newaxis,:]
        image8 = np.array(image8)[np.newaxis,:]
        image9 = np.array(image9)[np.newaxis,:]
        imagepool = [image0,image1,image2,image3,image4,image5,image6,image7,image8,image9]
        images_new = []
        labels = []
        for i in range(self.batch_size):
            index = random.randrange(0,10)
            images_new.append(imagepool[index])
            labels.append(index)
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        self.labels = labels

        image0 = [[1],[1],[1],[1],[1],[1],[0]]
        image1 = [[0],[1],[1],[0],[0],[0],[0]]
        image2 = [[1],[1],[0],[1],[1],[0],[1]]
        image3 = [[1],[1],[1],[1],[0],[0],[1]]
        image4 = [[0],[1],[1],[0],[0],[1],[1]]
        image5 = [[1],[0],[1],[1],[0],[1],[1]]
        image6 = [[1],[0],[1],[1],[1],[1],[1]]
        image7 = [[1],[1],[1],[0],[0],[0],[0]]
        image8 = [[1],[1],[1],[1],[1],[1],[1]]
        image9 = [[1],[1],[1],[1],[0],[1],[1]]
        image0 = np.array(image0)[np.newaxis,:]
        image1 = np.array(image1)[np.newaxis,:]
        image2 = np.array(image2)[np.newaxis,:]
        image3 = np.array(image3)[np.newaxis,:]
        image4 = np.array(image4)[np.newaxis,:]
        image5 = np.array(image5)[np.newaxis,:]
        image6 = np.array(image6)[np.newaxis,:]
        image7 = np.array(image7)[np.newaxis,:]
        image8 = np.array(image8)[np.newaxis,:]
        image9 = np.array(image9)[np.newaxis,:]
        imagepool = [image0,image1,image2,image3,image4,image5,image6,image7,image8,image9]
        images_new = []
        labels = []
        for i in range(self.batch_size):
            index = random.randrange(0,10)
            images_new.append(imagepool[index])
            labels.append(index)
        #print(labels)
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        self.labels = labels
"""