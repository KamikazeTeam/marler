import numpy as np
import agents, time
import torch,cv2,pygame
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
BLACK, ORANGE = pygame.color.THECOLORS['black'], pygame.color.THECOLORS['orange']
class SendInfo(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def getaction(self, obs, explore):
        act, act_info = self.agt.getaction(obs,explore)
        return act_info, act_info
class DrawInfo(agents.Wrapper):
    def getaction(self, obs, explore):
        act, act_info = self.agt.getaction(obs,explore)
        if self.drawinfo_weight_frequency!=0 and self.agt_step%self.drawinfo_weight_frequency==0:
            #print('drawing weight for agt_step:',self.agt_step)
            self.drawinfo_weight()
        if self.drawinfo_value_frequency!=0 and self.agt_step%self.drawinfo_value_frequency==0 or self.debug==True:
            for ienv in range(self.args.env_num):
                self.drawinfo_value(ienv,obs[ienv])
                self.debug = False
        self.agt_step+=1
        return act, act_info
    def __init__(self,agt,args,env):
        self.args,self.agt_step,self.debug = args,0,False
        self.frequency = [int(_) for _ in args.drawinfo.split(',')]
        self.drawinfo_value_frequency,self.drawinfo_weight_frequency = self.frequency[0],self.frequency[1]
        self.drawstatic_value_frequency,self.drawstatic_weight_frequency = self.frequency[2],self.frequency[3]
        agents.Wrapper.__init__(self,agt)
        if sum(self.frequency) != 0:
            model_children= list(self.agt.algo.model.base.children())+list(self.agt.algo.model.dist.children())
            self.layers  = []
            for layer in model_children:
                if type(layer) == nn.Sequential:
                    for layeri in layer:
                        self.layers.append(layeri)
                else:
                    self.layers.append(layer)
            print("Total layers:",len(self.layers),'(layer,weightshape,biasshape)')
            for i,layer in enumerate(self.layers):
                if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                    print(i,':',layer,layer.weight.shape,layer.bias.shape)
                else:
                    print(i,':',layer)
            self.figsize, dpi, self.roundnum = (args.infowidth,args.infoheight), args.infodpi, 1
            self.width, self.height = self.figsize[0]*dpi, self.figsize[1]*dpi
            self.vWriters = []
            for ienv in range(args.env_num):
                vWriters = []
                for ilayer in range(len(self.layers)+1):
                    videoname    = args.output_dir+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str(ienv)+'_'+str(ilayer)+'.mp4'
                    fps, fourcc  = args.fps, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
                    vWriters.append(cv2.VideoWriter(videoname, fourcc, fps, (self.width, self.height)))
                self.vWriters.append(vWriters)
            self.vWriters_static = []
            for ienv in range(args.env_num):
                vWriters_static = []
                for ilayer in range(len(self.layers)+1):
                    videoname    = args.output_dir+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str(ienv)+'_'+str(ilayer)+'_'+str('static')+'.mp4'
                    fps, fourcc  = args.fps, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
                    vWriters_static.append(cv2.VideoWriter(videoname, fourcc, fps, (640, 480)))
                self.vWriters_static.append(vWriters_static)
            self.wWriters = []
            for ilayer,layer in enumerate(self.layers):
                if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                    videoname    = args.output_dir+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str('w')+'_'+str(ilayer)+'.mp4'
                    fps, fourcc  = args.fps, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
                    self.wWriters.append(cv2.VideoWriter(videoname, fourcc, fps, (self.width, self.height)))
                else: self.wWriters.append(None)
            self.wWriters_static = []
            for ilayer,layer in enumerate(self.layers):
                if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                    videoname    = args.output_dir+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str('w')+'_'+str(ilayer)+'_'+str('static')+'.mp4'
                    fps, fourcc  = args.fps, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
                    self.wWriters_static.append(cv2.VideoWriter(videoname, fourcc, fps, (640, 480)))#(self.width, self.height)))
                else: self.wWriters_static.append(None)
            self.showframe  = False#True
            self.pygameflag = True
            if self.pygameflag:
                pygame.init()
                pygame.mixer.quit()
                self.screen = pygame.Surface((self.width, self.height))
                if self.showframe: 
                    self.display= pygame.display.set_mode([self.width, self.height])
            self.debug = True
    def __del__(self):
        if sum(self.frequency) != 0:
            for ienv in range(self.args.env_num):
                for ilayer in range(len(self.layers)+1):
                    self.vWriters[ienv][ilayer].release()
            for ienv in range(self.args.env_num):
                for ilayer in range(len(self.layers)+1):
                    self.vWriters_static[ienv][ilayer].release()
            for ilayer in range(len(self.layers)):
                if self.wWriters[ilayer]:
                    self.wWriters[ilayer].release()
            for ilayer in range(len(self.layers)):
                if self.wWriters_static[ilayer]:
                    self.wWriters_static[ilayer].release()
    def drawbyplt(self,data,xnum,ynum,i,fig,textflag=False):
        ax = fig.add_subplot(xnum,ynum,i+1)
        im = ax.imshow(data, cmap='gray') # cmap auto range
        if textflag:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(j,i,round(data[i,j],self.roundnum),ha="center",va="center",color="orange")
    def plttoframe(self,fig):
        fig.canvas.draw()# redraw the canvas
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')# convert canvas to image
        frame = frame.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)# frame is rgb, convert to opencv's default bgr
        return frame
    def drawbysurf(self,data,xnum,ynum,i,screen):
        data = data - data.min()
        if data.max()!=0: data = 255*data/data.max() # range to 0-255 # rerange data after reshape padding make 0 padding become gray
        size = [self.width//xnum, self.height//ynum]
        data_tiled = np.tile(data[:,:,np.newaxis],(1,1,3)).swapaxes(0,1)
        surf = pygame.surfarray.make_surface(data_tiled)
        surf = pygame.transform.scale(surf, size)
        screen.blit(surf,((i%xnum)*size[0],(i//xnum)*size[1]))
    def surftoframe(self,surf):
        frame = pygame.surfarray.array3d(surf).swapaxes(0,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    def wait_for_key_press(self):
        wait = True
        while wait:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    wait = False
                    break
    def drawinfo_i(self,layer_values,title=''):
        if self.debug: print(layer_values.shape)
        if len(layer_values.shape)==1:
            sqrt_len_layer_values = int(np.sqrt(len(layer_values)-1))+1
            layer_values_padded = torch.zeros([sqrt_len_layer_values*sqrt_len_layer_values])
            layer_values_padded[:len(layer_values)] = layer_values
            layer_values_show = layer_values_padded.reshape(1,sqrt_len_layer_values,sqrt_len_layer_values)
            if self.debug: print(layer_values_show.shape,'(reshaped)')
        else:
            layer_values_show = layer_values
        if not self.pygameflag:
            fig = plt.figure(figsize=self.figsize,tight_layout=True,num=title)
            #fig.suptitle(title)
        else: self.screen.fill(BLACK)
        sqrt_channel_num = int(np.sqrt(len(layer_values_show)-1))+1 # math.ceil
        if self.debug: print('sqrt_channel_num',sqrt_channel_num)
        for ichannel,layer_values_ichannel in enumerate(layer_values_show):
            data = layer_values_ichannel.detach().cpu().numpy()
            if not self.pygameflag: self.drawbyplt(data,sqrt_channel_num,sqrt_channel_num,ichannel,fig)
            else: self.drawbysurf(data,sqrt_channel_num,sqrt_channel_num,ichannel,self.screen)
        if not self.pygameflag:
            frame = None
            if self.showframe: plt.show()
            else: frame = self.plttoframe(fig)
            plt.close()
        else:
            fontsize, linespace, startline, vspace = 18, 18, 0, 0
            myfont = pygame.font.SysFont("monospace", fontsize)# font8=6px
            linestring = myfont.render('LAYER:%s' %str(title),True, ORANGE)
            self.screen.blit(linestring,(0,linespace*0+startline))
            for i in range(0,self.width, (self.width//sqrt_channel_num)):
                pygame.draw.line(self.screen, ORANGE, (i, 0), (i, self.height))
            for j in range(0,self.height,(self.height//sqrt_channel_num)):
                pygame.draw.line(self.screen, ORANGE, (0, j), (self.width, j))
            if self.showframe: 
                self.display.blit(self.screen,(0,0))
                pygame.display.update() #pygame.display.flip()
                self.wait_for_key_press()
            frame = self.surftoframe(self.screen)
        return frame
    def drawstatic_i(self,layer_values):
        fig = plt.figure(tight_layout=True)#,figsize=self.figsize)
        values = layer_values.detach().flatten().cpu().numpy()
        plt.hist(values, bins=20, density=True, histtype='step', facecolor="blue", edgecolor="blue", alpha=0.5)
        plt.ylim(0,10)
        plt.xlim(-1,1)
        frame = self.plttoframe(fig)
        plt.close()
        return frame        
    def drawinfo_value(self,ienv,obs):
        if self.debug: print('ienv:',ienv,'/',self.args.env_num)
        obs_tensor = torch.from_numpy(obs).float().cuda()
        obs_tensor = obs_tensor.permute(3,0,1,2) # go to channel,stack,width,height
        if obs_tensor.shape[1]==1: obs_tensor = obs_tensor.squeeze(1)
        else: obs_tensor = obs_tensor.squeeze(0)
        layer_values = obs_tensor
        frame = self.drawinfo_i(layer_values)
        self.vWriters[ienv][0].write(frame)
        if self.drawstatic_value_frequency!=0 and self.agt_step%self.drawstatic_value_frequency==0:
            frame = self.drawstatic_i(layer_values)
            self.vWriters_static[ienv][0].write(frame)
        for ilayer,layer in enumerate(self.layers[:-2]):
            if self.debug: print('ilayer:',ilayer,'/',len(self.layers[:-2]))
            layer_values = layer(torch.unsqueeze(layer_values,0))[0]
            frame = self.drawinfo_i(layer_values,title=str(ilayer)+':'+str((layer)))
            self.vWriters[ienv][ilayer+1].write(frame)
            if self.drawstatic_value_frequency!=0 and self.agt_step%self.drawstatic_value_frequency==0:
                frame = self.drawstatic_i(layer_values)
                self.vWriters_static[ienv][ilayer+1].write(frame)
        for ilayer,layer in enumerate(self.layers[-2:]):
            if self.debug: print('ilayer_head:',ilayer,'/',len(self.layers[-2:]))
            head_layer_values = layer(torch.unsqueeze(layer_values,0))[0]
            frame = self.drawinfo_i(head_layer_values)
            self.vWriters[ienv][ilayer-2].write(frame)
            if self.drawstatic_value_frequency!=0 and self.agt_step%self.drawstatic_value_frequency==0:
                frame = self.drawstatic_i(layer_values)
                self.vWriters_static[ienv][ilayer-2].write(frame)
    def drawinfo_weight(self):
        for ilayer,layer in enumerate(self.layers):
            if self.debug: print('ilayer_weight:',ilayer,'/',len(self.layers),'-',type(layer))
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                weight = layer.weight
                if self.debug: print(weight.shape)
                weight_reshaped = torch.flatten(weight,0,1)
                if self.debug: print(weight.shape,weight_reshaped.shape)
                frame = self.drawinfo_i(weight_reshaped,title=str(ilayer)+':'+str((layer)))
                self.wWriters[ilayer].write(frame)
                if self.drawstatic_weight_frequency!=0 and self.agt_step%self.drawstatic_weight_frequency==0:
                    frame = self.drawstatic_i(weight_reshaped)
                    self.wWriters_static[ilayer].write(frame)


