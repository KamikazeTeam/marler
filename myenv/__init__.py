from gym.envs.registration import register
register(id='shipS-v0',     entry_point='myenv.shipS:ShipS')
register(id='drone-v0',     entry_point='myenv.drone:Drone')
register(id='escape-v0',    entry_point='myenv.escape:Escape')
register(id='microa-v0',    entry_point='myenv.microa:Micro')
register(id='scan-v0',      entry_point='myenv.scan:Scan')
register(id='microaM-v0',   entry_point='myenv.microaM:MicroM')
register(id='scanM-v0',     entry_point='myenv.scanM:ScanM')
register(id='microaMs-v0',  entry_point='myenv.microaMs:MicroMs')
register(id='microaMsM-v0', entry_point='myenv.microaMsM:MicroMsM')
register(id='microMM-v0',   entry_point='myenv.microMM:MicroMM')
register(id='scanMM-v0',    entry_point='myenv.scanMM:ScanMM')
register(id='csc2-v0', entry_point='myenv.csc2:Csc2')
register(id='haliteM-v0', entry_point='myenv.haliteM:HaliteM')
register(id='CCA-v0', entry_point='myenv.CCA:CCA')
register(id='CCAX-v0', entry_point='myenv.CCAX:CCAX')
register(id='vision-v0', entry_point='myenv.vision:VISION')
register(id='moveMM-v0', entry_point='myenv.moveMM:MoveMM')
register(id='prices-v0', entry_point='myenv.prices:PRICES')
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
WHITE, BLACK = pygame.color.THECOLORS['white'], pygame.color.THECOLORS['black']
SGRAY = pygame.color.THECOLORS['gray8']
DGRAY, GRAY, BGRAY, GRAY169 = pygame.color.THECOLORS['gray15'], pygame.color.THECOLORS['gray45'], pygame.color.THECOLORS['gray75'], pygame.color.THECOLORS['darkgray']#38,115,191
RED, GREEN, BLUE = pygame.color.THECOLORS['red'], pygame.color.THECOLORS['green'], pygame.color.THECOLORS['blue']
YELLOW, MAGENTA, CYAN = pygame.color.THECOLORS['yellow'], pygame.color.THECOLORS['magenta'], pygame.color.THECOLORS['cyan']
#print('RED',RED,'GREEN',GREEN,'BLUE',BLUE,'YELLOW',YELLOW,'MAGENTA',MAGENTA,'CYAN',CYAN)
class LPVSprite(pygame.sprite.Sprite):#DirtySprite
    def __init__(self, ispecies=0, inum=0, color=WHITE, lenx=1, leny=1, posx=0, posy=0, velx=0, vely=0, hp_max=1, hp_crt=1, atk=1, rng=1):
        super().__init__()
        self.hitten_list = []
        self.ispecies, self.inum, self.color = ispecies, inum, color
        self.lenx, self.leny, self.posx, self.posy, self.velx, self.vely, self.pposx, self.pposy = lenx, leny, posx, posy, velx, vely, posx, posy
        self.gain, self.loss, self.live, self.hp_max, self.hp_crt, self.atk, self.rng = 0, 0, 1, hp_max, hp_crt, atk, rng
        self.image = pygame.Surface([int(lenx), int(leny)])
        self.image.fill(color)
        self.rect  = self.image.get_rect(center=(int(posx), int(posy)))
        self.rect_hit = self.rect.inflate(rng*2, rng*2)
    def vel_set(self,velx,vely):
        self.velx, self.vely = velx, vely
    def stepback(self):
        self.posx, self.posy = self.pposx, self.pposy
        self.rect.center = (int(self.posx), int(self.posy))
        self.rect_hit.center = self.rect.center
    def update(self):
        self.pposx, self.pposy = self.posx, self.posy
        self.posx += self.velx
        self.posy += self.vely
        self.rect.center = (int(self.posx), int(self.posy))
        self.rect_hit.center = self.rect.center
    def pos_move(self,x_diff,y_diff):
        self.pposx, self.pposy = self.posx, self.posy
        self.posx += x_diff
        self.posy += y_diff
        self.rect.center = (int(self.posx), int(self.posy))
        self.rect_hit.center = self.rect.center
    def pos_clip(self,x_min,x_max,y_min,y_max):
        if self.rect.x < x_min:
            self.rect.x = x_min
            self.posx = self.rect.centerx
        if self.rect.x+self.lenx-1 > x_max:
            self.rect.x = x_max-self.lenx+1
            self.posx = self.rect.centerx
        if self.rect.y < y_min:
            self.rect.y = y_min
            self.posy = self.rect.centery
        if self.rect.y+self.leny-1 > y_max:
            self.rect.y = y_max-self.leny+1
            self.posy = self.rect.centery
        if self.rect_hit.center == self.rect.center: return False
        self.rect_hit.center = self.rect.center
        return True
def hit_collision(sprite1, sprite2):
    return sprite1.rect_hit.colliderect(sprite2.rect)
