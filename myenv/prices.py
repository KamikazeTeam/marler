import numpy as np
import os, random, gym, gym.spaces, json, easydict, time, cv2, scipy, talib
from itertools import cycle
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from myenv import *
import torch,torchvision
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
import pandas as pd
from myenv.pricesfunc import debug,getdatetimestring,getdata,filldata,expanddata,groupdata,logdata,getpositions,calsharpes, indicators,statics,getsignals,drawcurves
from itertools import cycle
COLORS = ['red','orange','green','cyan','blue','purple']#,'black']
MARKERS= ['+', '.', 'o', '*']#",",".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","d","D","|","_","","","","",""
def drawdistribution(df,column):
    data = np.asarray(df[column].fillna(0))
    print(format(column,'8s'),data.shape,np.mean(data),np.std(data),np.min(data),np.max(data))
    plt.figure(column)
    plt.hist(data.flatten(),bins=200,range=(-0.005,0.005))#range(min(data), max(data) + binwidth, binwidth))#'auto')
    plt.axvline(x= 0.001,color='red')
    plt.axvline(x=-0.002,color='red')
    plt.axvline(x= uplimit,color='blue')
    plt.axvline(x= dnlimit,color='blue')
from torch.utils.data.dataset import Dataset
class DatasetFromDataFrame(Dataset):
    def __init__(self, dataframe, transform=None):
        df = dataframe.copy()#[['open','close','high','low','amount']].copy() 
        # need copy to tell pandas you want a refer to modify or a copy to use other way
        for column in df.columns:
            if column=='time': continue
            df[column] = (df[column]-np.mean(df[column]))/np.std(df[column])


        self.backward, forward = 4,3#79, 3
        self.size,self.padding = [5,5],0#[20,20],[(6,6),(6,6)]#up down left right
        obscolumns = ['open','close','high','low','amount']
        for i in range(1,self.backward+1):
            df[str(i)+'open']  = df['open'].shift(periods=i,fill_value=0)
            df[str(i)+'close'] = df['close'].shift(periods=i,fill_value=0)
            df[str(i)+'high']  = df['high'].shift(periods=i,fill_value=0)
            df[str(i)+'low']   = df['low'].shift(periods=i,fill_value=0)
            df[str(i)+'amount']= df['amount'].shift(periods=i,fill_value=0)
            obscolumns = obscolumns+[str(i)+'open',str(i)+'close',str(i)+'high',str(i)+'low',str(i)+'amount']
        #debug(df,20)
        for i in range(1,forward+1):
            df[str(i)+'r'] = df['close'].diff(periods=i).shift(periods=-i,fill_value=0)
        df['up']  = ((df['1r']>0)&(df['2r']>0)&(df['3r']>0)).astype(int)
        df['down']=-((df['1r']<0)&(df['2r']<0)&(df['3r']<0)).astype(int)
        df['signal'] = df['up']+df['down']
        df['signal']+= 1
        df = df[self.backward:-forward]
        # for test
        df['signal'] = df['signal'].shift(periods=3,fill_value=0)
        df = df[3:]
        # debug(df,10)
        # for column in ['open','close','high','low','amount','count','signal']:
        #     drawdistribution(df,column)


        self.data  = np.asarray(df[obscolumns])
        self.labels= np.asarray(df['signal'])
        print('data   shape',self.data.shape)
        print('labels shape',self.labels.shape)
        self.transform = transform
        self.df = df

        if 1: # linear test
            data = np.array(range(len(df)))[:,np.newaxis]
            self.data = np.concatenate((data, data+1), axis=1)
            pads = np.zeros(shape=(len(df),self.size[0]*self.size[1]-2))
            self.data = np.concatenate((self.data, pads), axis=1)
            self.labels = (data*2+1).squeeze()
            # print(self.labels.shape)
            # print(self.data.shape)
            # exit()
    def __getitem__(self, index):
        img_label = self.labels[index]




        img_as_np = self.data[index].reshape(self.size)#.astype('uint8')
        img_as_np = np.pad(img_as_np,self.padding,'constant')
        img_as_img= img_as_np#Image.fromarray(img_as_np)
        # img_as_img= img_as_img.convert('L')
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        return (img_as_tensor, img_label)
    def __len__(self):
        return self.labels.shape[0]
class TimeLineFigure:
    def __init__(self,datalength,datainterval,datatime,dataindex,marker):
        x_max,x_ticks = datalength,datalength//50
        print('x_max,x_ticks:',x_max,x_ticks)
        periods = [1,5,10,60,240,1440,10080]
        for period in periods:
            if datalength//period>50: continue
            else: break
        x_ticks = period
        x_max = ((datalength-1)//x_ticks+1)*x_ticks
        print('x_max,x_ticks:',x_max,x_ticks)
        self.x_max,self.x_ticks = x_max,x_ticks
        self.datainterval = datainterval
        self.datatime,self.dataindex = datatime,dataindex
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['grid.color']      = 'black'
        mpl.rcParams['grid.linewidth']  = .1
        mpl.rcParams['lines.linewidth'] = .3
        mpl.rcParams['lines.linestyle'] = '-'
        if marker:
            mpl.rcParams['lines.marker']    = '.'
            mpl.rcParams['lines.markersize']= 1
    def figure(self,name=''):
        fig = plt.figure(name,figsize=(16,9))
        plt.grid()
        plt.xticks(rotation=-90)#,Fontsize=6)#10
        plt.gca().set_xticks([i+self.dataindex[0] for i in range(0,self.x_max,self.x_ticks)])
        plt.gca().set_xticklabels([self.datatime[self.dataindex[i]] for i in range(0,self.x_max,self.x_ticks)])
        return fig
def addreference(data,indicator,instrument_id):
    #lines = open('reference/'+indicator+instrument_id+'.csv').readlines()
    lines = open('reference/Funding Rate History_'+instrument_id.split('_')[-1].replace('-','')+' Perpetual_2022-06-25'+'.csv').readlines()
    timestamp,fundingrate = [],[]
    for line in lines[1:]:
        elements = line.split(',')
        timestamp.append(elements[0][1:-1])
        fundingrate.append(float(elements[3][1:-3])/100)
    data_ref = pd.DataFrame({'time':timestamp,indicator+instrument_id:fundingrate})
    data_ref['time'] = pd.to_datetime(data_ref['time'],utc=True)
    data = data.merge(data_ref,how='left',on='time')
    data[indicator+instrument_id] = data[indicator+instrument_id].interpolate(method='bfill').fillna(0.0)#,limit=1)#ffill
    return data
def getfillgrouplogdata(datetime,instrument_id):
    tags = 'open,close,high,low,volume,count,amount'
    tags = tags+',volume_long,count_long,amount_long,volume_base'
    # instrument_id= 'BINANCE_SWAP_ETH-USDT'
    measurements = 'bfs.autogen.klinehistory'
    headdatetime = getdatetimestring(datetime[0]+' +0000')
    taildatetime = getdatetimestring(datetime[1]+' +0000')
    querycommand = 'select '+tags+' from '+measurements+' where '+'time<='+taildatetime+' and '+'time>='+headdatetime \
                    +' and '+"instrument_id='"+instrument_id+"'"
    data = getdata(host='47.57.72.27', port=8086, querycommand=querycommand)
    # debug(data,30)
    # exit()
    # debug(data,500)
    # dftmp = data.duplicated(subset='time')
    # debug(dftmp,500)
    data = data.interpolate(method='bfill',limit=1)
    # debug(data,500)
    # exit()
    data = data.drop_duplicates(subset='time') # there are some lines with same time value and first line without but second line with amount, count_long, amount_long values
    data = filldata(data,datetime[0],datetime[1])
    # debug(data,500)
    # exit()
    data = expanddata(data,datetime[0],datetime[1],datetime[2])
    data = groupdata(data,length=int(datetime[3]))
    data = logdata(data)
    data = indicators(data,indicator='EMA',indexes=[7,15,30])
    return data
def getzerotimevalue(data,column): # ensure zerotime value not NaN
    return data[data[column].isnull()==False][column].iloc[0]
def drawstrategycurves(data,fig,instrument_ids,strategyname):
    colors = cycle(COLORS)
    for instrument_id in instrument_ids:
        color = next(colors)
        plt.plot(data[openpricename+instrument_id]-getzerotimevalue(data,openpricename+instrument_id.replace('SPOT','SWAP')),color=color) #align spot&swap
    plt.plot(data[strategyname+'-total']*1,color='black') # draw trade curve
    for instrument_id in instrument_ids: # draw trade marks
        prefix = strategyname+'long'+instrument_id
        if prefix+'-openprice' and prefix+'-closeprice' in data.columns:
            plt.scatter( data[data[prefix+'-openprice']!=0].index
                        ,data[data[prefix+'-openprice']!=0][openpricename+instrument_id]-getzerotimevalue(data,openpricename+instrument_id)
                        ,color='red', marker='+', s=35)
            plt.scatter( data[data[prefix+'-closeprice']!=0].index
                        ,data[data[prefix+'-closeprice']!=0][openpricename+instrument_id]-getzerotimevalue(data,openpricename+instrument_id)
                        ,color='red', marker='.', s=35)
            print('Draw:',prefix)
            debug(data[data[prefix+'-openprice']!=0]['time'],20)
            debug(data[data[prefix+'-closeprice']!=0]['time'],20)
        prefix = strategyname+'short'+instrument_id
        if prefix+'-openprice' and prefix+'-closeprice' in data.columns:
            plt.scatter( data[data[prefix+'-openprice']!=0].index
                        ,data[data[prefix+'-openprice']!=0][openpricename+instrument_id]-getzerotimevalue(data,openpricename+instrument_id)
                        ,color='blue', marker='+', s=35)
            plt.scatter( data[data[prefix+'-closeprice']!=0].index
                        ,data[data[prefix+'-closeprice']!=0][openpricename+instrument_id]-getzerotimevalue(data,openpricename+instrument_id)
                        ,color='blue', marker='.', s=35)
            print('Draw:',prefix)
            debug(data[data[prefix+'-openprice']!=0]['time'],20)
            debug(data[data[prefix+'-closeprice']!=0]['time'],20)
pairs = 'ETH-USDT'#ADA,BCH,BNB,BTC,DOT,EOS,ETC,ETH,FIL,LINK,LTC,TRX,XRP,#AVAX
strategyname,target = 'bullshort',{}
target['random']     = {'SPOT':'BINANCE_SPOT_'+pairs,'SWAP':'BINANCE_SWAP_'+pairs}
target['arbitrage']  = {'SPOT':'BINANCE_SPOT_'+pairs,'SWAP':'BINANCE_SWAP_'+pairs}
target['valueinvest']= {'SPOT':'BINANCE_SPOT_'+pairs,'SWAP':'BINANCE_SWAP_'+pairs}
target['momentum']   = {'SPOT':'BINANCE_SPOT_'+pairs,'SWAP':'BINANCE_SWAP_'+pairs}
target['bullshort']  = {'SPOT':'BINANCE_SPOT_'+pairs,'SWAP':'BINANCE_SWAP_'+pairs}
openpricename,closepricename,holdpricename='open','open','open'
uplimit,dnlimit = -0.0009,-0.0012#-0.0003,-0.0028
riskbar = 0.02#105
fee = 0.000#5#(0.00075+0.00036)/2#0.001+0.0005 # why each trade earn 0.002 not make all capital gain equal zero?
def getstrategypositions(data,tlf,strategyname):
    if strategyname == 'random':
        data[strategyname+'-tradesignal'] = pd.Series(np.random.randint(low=-1, high=2, size=len(data)),index=data.index)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= 0.0)
        data = getpositions(data,tradesignal=-data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='short',target=target[strategyname]
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-0.0)
        data[strategyname+'-total'] = data[strategyname+'long'+target[strategyname]+'-total']-data[strategyname+'short'+target[strategyname]+'-total']
    if strategyname == 'arbitrage':
        data['split'] = data[openpricename+target[strategyname]['SPOT']]-data[openpricename+target[strategyname]['SWAP']]
        data[strategyname+'-opensignal']  = ((data['split']> uplimit) # sell spot buy swap
                                            &(data['split']< riskbar)
                                            ).astype(int)
        data[strategyname+'-closesignal'] =-((data['split']< dnlimit)
                                            |(data['split'].shift(periods=-1,fill_value=0)> riskbar)
                                            |(data['split'].shift(periods=-1,fill_value=0)<-riskbar)
                                            ).astype(int)
        data[strategyname+'-tradesignal'] = (data[strategyname+'-opensignal']+data[strategyname+'-closesignal']*2).clip(lower=-1) # to avoid do nothing while open and close by shift condition
        data[strategyname+'-tradesignal'][data[openpricename+target[strategyname]['SWAP']].isnull()]=0
        data[strategyname+'-tradesignal'][data[openpricename+target[strategyname]['SPOT']].isnull()]=0 # to avoid half open or close by one data missing
        data[strategyname+'-tradesignal'][data[closepricename+target[strategyname]['SWAP']].isnull()]=0
        data[strategyname+'-tradesignal'][data[closepricename+target[strategyname]['SPOT']].isnull()]=0
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= fee)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='short',target=target[strategyname]['SPOT']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)
        data[strategyname+'-opensignal2'] = ((data['split']< dnlimit) # buy spot sell swap
                                            &(data['split']>-riskbar)
                                            ).astype(int)
        data[strategyname+'-closesignal2']=-((data['split']> uplimit)
                                            |(data['split'].shift(periods=-1,fill_value=0)> riskbar)
                                            |(data['split'].shift(periods=-1,fill_value=0)<-riskbar)
                                            ).astype(int)
        data[strategyname+'-tradesignal2']= (data[strategyname+'-opensignal2']+data[strategyname+'-closesignal2']*2).clip(lower=-1)
        data[strategyname+'-tradesignal2'][data[openpricename+target[strategyname]['SWAP']].isnull()]=0
        data[strategyname+'-tradesignal2'][data[openpricename+target[strategyname]['SPOT']].isnull()]=0
        data[strategyname+'-tradesignal2'][data[closepricename+target[strategyname]['SWAP']].isnull()]=0
        data[strategyname+'-tradesignal2'][data[closepricename+target[strategyname]['SPOT']].isnull()]=0
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal2']
                            ,strategyname=strategyname,direction='short',target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal2']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]['SPOT']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= fee)
        data[strategyname+'-total'] = \
                                    +data[strategyname+'long' +target[strategyname]['SWAP']+'-total'] \
                                    -data[strategyname+'short'+target[strategyname]['SPOT']+'-total'] \
                                    -data[strategyname+'short'+target[strategyname]['SWAP']+'-total'] \
                                    +data[strategyname+'long' +target[strategyname]['SPOT']+'-total']
    if strategyname == 'valueinvest':
        # data[strategyname+'-closesignal'] =-(((-data['fundingrate'+target[strategyname]['SWAP']]>=-0.0001))
        #                                     #|(-data['fundingrate'+target[strategyname]['SWAP']]> 0.0005))
        #                                     #|(data['avg_volumeEMA120']<data['lower'])
        #                                     ).astype(int)
        # data[strategyname+'-opensignal']  = ((-data['fundingrate'+target[strategyname]['SWAP']]<-0.0001)
        #                                     #&(data['momentum']>0)
        #                                     #&(data['avg_volumeEMA120']>data['upper'])
        #                                     ).astype(int)#.shift(periods=1,fill_value=0)
        data[strategyname+'-opensignal']  = (((-data['fundingrate'+target[strategyname]['SWAP']]>-0.0001)
                                             &(-data['fundingrate'+target[strategyname]['SWAP']]< 0.0005))
                                            #|(data['avg_volumeEMA120']<data['lower'])
                                            ).astype(int)
        data[strategyname+'-closesignal'] =-((-data['fundingrate'+target[strategyname]['SWAP']]<-0.0001)
                                            #&(data['momentum']>0)
                                            &(data['avg_volumeEMA120']>data['upper'])
                                            ).astype(int)
        data[strategyname+'-tradesignal'] = ((data[strategyname+'-opensignal']+data[strategyname+'-closesignal'])).shift(periods=0,fill_value=0)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= fee)
        data = getpositions(data,tradesignal=-data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='short', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)
        data[strategyname+'-total'] = \
                                    +data[strategyname+'long' +target[strategyname]['SWAP']+'-total'] \
                                    -data[strategyname+'short'+target[strategyname]['SWAP']+'-total']
    if strategyname == 'momentum':
        data[strategyname+'-opensignal']  = (data['momentum']> 0.0).astype(int)
        data[strategyname+'-closesignal'] =-(data['momentum']< 0.0).astype(int)
        data[strategyname+'-tradesignal'] = (data[strategyname+'-opensignal']+data[strategyname+'-closesignal'])
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= fee)
        data = getpositions(data,tradesignal=-data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='short', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)
        data[strategyname+'-total'] = \
                                    +data[strategyname+'long' +target[strategyname]['SWAP']+'-total'] \
                                    -data[strategyname+'short'+target[strategyname]['SWAP']+'-total']
    if strategyname == 'bullshort':
        data[strategyname+'-opensignal']  = (((-data['fundingrate'+target[strategyname]['SWAP']]> 0.0001)
                                             &(-data['fundingrate'+target[strategyname]['SWAP']]< 0.0005))
                                            #|(data['avg_volumeEMA120']<data['lower'])
                                            ).astype(int)
        data[strategyname+'-closesignal'] =-((-data['fundingrate'+target[strategyname]['SWAP']]<-0.0001)
                                            #&(data['momentumpriceMA60']<0)
                                            #&(data['avg_volumeEMA120']>data['upper'])
                                            ).astype(int)
        data[strategyname+'-tradesignal'] = ((data[strategyname+'-opensignal']+data[strategyname+'-closesignal'])).shift(periods=0,fill_value=0)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal']
                            ,strategyname=strategyname,direction='long', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee= fee)


        data[strategyname+'-opensignal2'] = ((-data['fundingrate'+target[strategyname]['SWAP']]<-0.0001)
                                            #&(data['momentumpriceMA60']>0)
                                            #&(data['avg_volumeEMA120']>data['upper'])
                                            ).astype(int)
        data[strategyname+'-closesignal2']=-(((-data['fundingrate'+target[strategyname]['SWAP']]>=-0.0001))
                                            #&(-data['fundingrate'+target[strategyname]['SWAP']]<= 0.0001))
                                            #|(data['avg_volumeEMA120']<data['lower'])
                                            ).astype(int)
        data[strategyname+'-tradesignal2']= ((data[strategyname+'-opensignal2']+data[strategyname+'-closesignal2'])).shift(periods=0,fill_value=0)
        data = getpositions(data,tradesignal= data[strategyname+'-tradesignal2']
                            ,strategyname=strategyname,direction='short', target=target[strategyname]['SWAP']
                            ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)


        # data[strategyname+'-opensignal2'] = ((-data['fundingrate'+target[strategyname]['SWAP']]<-0.0001)
        #                                     #&(data['momentumpriceMA60']>0)
        #                                     #&(data['avg_volumeEMA120']>data['upper'])
        #                                     ).astype(int)
        # data[strategyname+'-closesignal2']=-(((-data['fundingrate'+target[strategyname]['SWAP']]>=-0.0001)
        #                                      &(-data['fundingrate'+target[strategyname]['SWAP']]<= 0.0001))
        #                                     #|(data['avg_volumeEMA120']<data['lower'])
        #                                     ).astype(int)
        # data[strategyname+'-tradesignal2']= ((data[strategyname+'-opensignal2']+data[strategyname+'-closesignal2'])).shift(periods=0,fill_value=0)
        # data = getpositions(data,tradesignal= data[strategyname+'-tradesignal2']
        #                     ,strategyname=strategyname,direction='short', target=target[strategyname]['SWAP']
        #                     ,openpricename=openpricename,closepricename=closepricename,holdpricename=holdpricename,fee=-fee)


        data[strategyname+'-total'] = \
                                    +data[strategyname+'long' +target[strategyname]['SWAP']+'-total'] \
                                    +data[strategyname+'short'+target[strategyname]['SWAP']+'-total']

    # debug(-data['fundingrate'+target[strategyname]['SWAP']],100)
    # debug(data[strategyname+'-closesignal2'],100)

    stdearn = data[openpricename].diff().fillna(0) # print sortino ratio # use first instrument as standard benchmark
    profits = data[strategyname+'-total'].diff().fillna(0)
    benchmarks_sortino,strategies_sortino=calsharpes(stdearn,profits,525600/tlf.datainterval)
    stdearn = np.power(np.e,data[openpricename].diff().fillna(0))-1
    profits = np.power(np.e,data[strategyname+'-total'].diff().fillna(0))-1
    benchmarks_sortino,strategies_sortino=calsharpes(stdearn,profits,525600/tlf.datainterval)
    return data,str(format(benchmarks_sortino,'+.3f'))+str(format(strategies_sortino,'+.3f'))
# def normalize01(data,minvalue,maxvalue):
#     return (data-minvalue)/(maxvalue-minvalue)
def drawEMA(data,column,lengths,offset=None,ratio=None,color=None,linestyle='-',drawratio=1,drawoffset=0):
    colors = cycle(COLORS)
    if offset==None:offset= data[column].min()
    if ratio==None: ratio = 1/(data[column].max()-data[column].min())
    if color==None: drawcolor = next(colors)
    else:           drawcolor = color
    plt.plot((data[column]-offset)*ratio/drawratio+drawoffset,color=drawcolor,linestyle=linestyle,alpha=0.32)
    for length in lengths:
        data[column+'EMA'+str(length)] = talib.EMA(data[column],length)
        data[column+'EMA'+str(length)] = data[column+'EMA'+str(length)].fillna(getzerotimevalue(data,column+'EMA'+str(length)))
        if color==None: drawcolor = next(colors)
        else:           drawcolor = color
        plt.plot((data[column+'EMA'+str(length)]-offset)*ratio/drawratio+drawoffset,color=drawcolor,linestyle=linestyle)
class PRICES(gym.Env):
    def __init__(self):
        super().__init__()
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = easydict.EasyDict(json.load(envinfo_file))
        self.args = envinfo_args_dict
        self.figfolder = './figures/'
        os.makedirs(self.figfolder,exist_ok=True)
        #datetime  = '2020-06-01 01:00:00,2022-02-01 13:00:00,1min,120'.split(',')#self.args.env_type.split(',')
        datetimestrings=[
                        #'2020-06-01 01:00:00,2020-07-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2020-08-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2020-09-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2020-10-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2020-11-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2020-12-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-01-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-02-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-03-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-04-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-05-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-06-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-07-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-08-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-09-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-10-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-11-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2021-12-01 17:00:00,1min,120',
                        # '2020-06-01 01:00:00,2022-01-01 17:00:00,1min,120',
                        #'2022-03-10 07:00:00,2022-03-10 11:00:00,1min,120',
                        '2022-03-01 01:00:00,2022-06-01 17:00:00,1min,120',
                        ]
        for i,datetimestring in enumerate(datetimestrings):
            datetime = datetimestring.split(',')
            instrument_ids = ['BINANCE_SPOT_'+pairs,'BINANCE_SWAP_'+pairs]#'BINANCE_QUARTER_','BINANCE_NEXT_QUARTER_'# refer listed
            #instrument_ids = ['BINANCE_SWAP_'+pairs,'BINANCE_SPOT_'+pairs]
            data = getfillgrouplogdata(datetime,instrument_ids[0])
            # debug(data,200)
            # exit()
            for instrument_id in instrument_ids:
                data_ref = getfillgrouplogdata(datetime,instrument_id)
                # debug(data_ref,-10)
                data = data.merge(data_ref,how='left',on='time',suffixes=['',instrument_id])#outer
                # debug(data,-10)
            # exit()
            data = addreference(data,'fundingrate',target[strategyname]['SWAP'])
            self.instrument_ids = instrument_ids # target and refer data combined
            self.data_org = data
            # debug(self.data_org,-5)
            # exit()
            self.tlf = TimeLineFigure(len(data),int(datetime[3]),data['time'],data.index,marker=False)#True False
            fig = self.tlf.figure('splitcurve'+str(i))
            # data['split'] = data[openpricename+target[strategyname]['SPOT']]-data[openpricename+target[strategyname]['SWAP']]
            # plt.plot(data['split'],color='black',alpha=0.2)
            # colors = cycle(COLORS)
            # for index in [36,120,360,1200]:
            #     data['splitEMA'+str(index)] = talib.EMA(data['split'],index).fillna(0)
            #     color = next(colors)
            #     plt.plot(data['splitEMA'+str(index)],color=color)
            plt.plot(-data['fundingrate'+target[strategyname]['SWAP']]/10,color='black',alpha=0.2)
            plt.plot((data[openpricename+target[strategyname]['SWAP']]-getzerotimevalue(data,openpricename+target[strategyname]['SWAP']))/1000,color='black')
            data['diff'] = data[openpricename+target[strategyname]['SWAP']].diff().fillna(0)
            # plt.plot( data['diff']/1000,color='black',alpha=0.5)
            plt.ylim([-0.001,0.003])
            # exit()
            # plt.axhline(y= 0.001,color='red')
            # plt.axhline(y=-0.002,color='red')
            # plt.axhline(y= uplimit,color='blue')
            # plt.axhline(y= dnlimit,color='blue')
            plt.fill_between(data.index,-0.001,-0.0001,where=-data['fundingrate'+target[strategyname]['SWAP']]>-0.0000,color='green',alpha=0.051)
            plt.fill_between(data.index,-0.001,-0.0001,where=-data['fundingrate'+target[strategyname]['SWAP']]<-0.0002,color='red',alpha=0.051)
            # plt.fill_between(data.index,-0.002,-0.001,where=-data['fundingrate'+target[strategyname]['SWAP']]> 0.0001,color='green',alpha=0.051)
            # plt.fill_between(data.index,-0.002,-0.001,where=-data['fundingrate'+target[strategyname]['SWAP']]<-0.0003,color='red',alpha=0.051)
            # drawdistribution(data,'split')
            data['hldiff'] = data['high']-data['low']

            emaperiod = 12
            data['hldiff'] = talib.EMA(data['hldiff'],emaperiod)
            data['hldiff'] = data['hldiff'].fillna(getzerotimevalue(data,'hldiff'))
            data['VT'] = talib.EMA(data['VT'],emaperiod)
            data['VT'] = data['VT'].fillna(getzerotimevalue(data,'VT'))

            data['PD'] = data['ER'].copy()
            # data = data[:100]
            # debug(data['ER'])
            datas = []
            for i in range(12):
                meani = data['ER'][i::12].mean()
                # print(meani)
                datai = data['ER'][i::12]-meani
                datas.append(datai)
            # print(datas)
            # print(datas[0])
            # print(len(datas[0]))
            records = []
            for j in range(len(datas[0])):
                for i in range(12):
                    try:
                        # print(i,j,datas[i].iloc[j])
                        records.append(datas[i].iloc[j])
                    except:
                        # print(i,j)
                        break
            df = pd.Series(records)
            data['ER'] = df
            # debug(df)
            # exit()
            print(data['ER'].describe())
            print(data['PD'].describe())

            # emaperiod = 60
            # data['ER'] = talib.EMA(data['ER'],emaperiod)
            # data['ER'] = data['ER'].fillna(getzerotimevalue(data,'ER'))
            # data['PD'] = talib.EMA(data['PD'],emaperiod)
            # data['PD'] = data['PD'].fillna(getzerotimevalue(data,'PD'))

            plt.plot(data['ER']/1000*4+0.002,color='red',alpha=0.5,label=pairs+':efficient ratio|zoom=/1000*4')
            plt.plot(data['PD']/1000*4+0.001,color='green',alpha=0.5,label='1/(price density)|zoom=/1000*4')
            plt.plot(data['VT']/10,color='blue',alpha=0.5,label='(std of return)/min|zoom=/10')
            plt.plot(data['hldiff']/100-0.001,color='purple',alpha=0.5,label='ln(high)-ln(low)|zoom=/100')
            plt.legend(loc="upper left")
            qat = data['VT'].quantile([0.1, 0.5, 0.9])
            plt.fill_between(data.index,0.000,0.001,where=data['VT']>qat.iloc[-1],color='red',alpha=0.051)
            plt.fill_between(data.index,0.000,0.001,where=data['VT']<qat.iloc[0],color='green',alpha=0.051)
            qat = data['PD'].quantile([0.1, 0.5, 0.9])
            plt.fill_between(data.index,0.001,0.002,where=data['PD']>qat.iloc[-1],color='red',alpha=0.051)
            plt.fill_between(data.index,0.001,0.002,where=data['PD']<qat.iloc[0],color='green',alpha=0.051)
            qat = data['ER'].quantile([0.1, 0.5, 0.9])
            plt.fill_between(data.index,0.002,0.003,where=data['ER']>qat.iloc[-1],color='red',alpha=0.051)
            plt.fill_between(data.index,0.002,0.003,where=data['ER']<qat.iloc[0],color='green',alpha=0.051)

            # lengths = [4,12]
            # drawEMA(data,'volume',lengths,linestyle='-' ,drawratio=1000,drawoffset=0.002)
            # drawEMA(data,'count', lengths,linestyle='--',drawratio=1000,drawoffset=0.002)
            # # drawEMA(data,'amount_long',lengths,linestyle=':',drawratio=500)
            # data['long_ratio'] = data['amount_long']/data['volume']
            # drawEMA(data,'long_ratio',lengths,color=None,linestyle='-.',drawratio=1000,drawoffset=-0.003)
            # data['avg_volume'] = data['volume']/data['count']
            # data['avg_volume'] = np.log(data['avg_volume'])
            # drawEMA(data,'avg_volume',lengths,color=None,linestyle='-', drawratio=500)

            # lengths = [4]
            # for length in lengths:
            #     # data['avg_volumeEMA'+str(length)] = talib.EMA(data['volume']/data['count'],length) # avg around 0.04, if value too small BBANDS cal error...
            #     # # print('avg mean',(data['avg_volumeEMA'+str(length)].fillna(0).mean()))
            #     # data['avg_volumeEMA'+str(length)] = data['avg_volumeEMA'+str(length)].fillna(getzerotimevalue(data,'avg_volumeEMA'+str(length)))
            #     # # color = next(colors)
            #     # maxvalue = data['avg_volumeEMA'+str(length)].max()
            #     # minvalue = data['avg_volumeEMA'+str(length)].min()
            #     # print(maxvalue,minvalue)
            #     # plt.plot((data['avg_volumeEMA'+str(length)]-minvalue)/(maxvalue-minvalue)/500,color='blue',linestyle='-')
            #     column = 'avg_volume'
            #     data['upper'],data['middle'],data['lower'] = talib.BBANDS(data['avg_volume'],matype=talib.MA_Type.EMA,timeperiod=5,nbdevup=2,nbdevdn=2)
            #     dataoffset = data[column+'EMA'+str(length)].min()
            #     dataratio  = 1/(data[column+'EMA'+str(length)].max()-data[column+'EMA'+str(length)].min())
            #     upper= (data['upper'] -dataoffset)*dataratio/drawratio
            #     lower= (data['lower'] -dataoffset)*dataratio/drawratio
            #     middle=(data['middle']-dataoffset)*dataratio/drawratio
            #     plt.fill_between(x=data.index,y1=lower,y2=upper,color='blue',alpha=0.051)
            #     plt.plot(middle,color='blue',linestyle='--')
            # count_mean= data['count'].rolling(120).mean().fillna(0)
            # count_std = data['count'].rolling(120).std().fillna(0)
            # plt.plot(count_mean/1000000000*3*3,color='blue',linestyle='--')
            # plt.plot(count_std/100000000,color='blue')
            # debug(count_std,200)

                # ratioorg = (data[column+'EMA'+str(length)]-dataoffset)*dataratio/500
                # #data['momentum'] = talib.MOM(ratioorg,timeperiod=10)
                # data['momentum'] = talib.MOM(data['middle'],timeperiod=10)
                # plotdatamomentum = (data['momentum']-data['momentum'].min())/(data['momentum'].max()-data['momentum'].min())
                # plotdatamomentum = (plotdatamomentum-0.5)*2
                # plt.plot(plotdatamomentum/1000,color='blue',linestyle='--')

                # ratioorg = (data[column+'EMA'+str(length)]-dataoffset)*dataratio/drawratio
                # #data['momentum'] = talib.MOM(ratioorg,timeperiod=10)
                # data['momentum'] = talib.MOM(middle,timeperiod=120)
                # #data['momentumMA']=talib.EMA(data['momentum'],120)
                # #data['momentumMA']=data['momentumMA'].fillna(getzerotimevalue(data,'momentumMA'))
                # plt.plot(data['momentum']  *5+0.001,color='blue',linestyle='--')
                # #plt.plot(data['momentumMA']*5+0.001,color='blue',linestyle=':')

            # length = '12'
            # # data['pricemomentum'] = talib.MOM(data[openpricename+target[strategyname]['SWAP']],timeperiod=10)
            # # plt.plot(data['pricemomentum']/1000,color='black',linestyle='--',alpha=0.52)
            # data['priceMA'+length] = talib.EMA(data[openpricename+target[strategyname]['SWAP']],int(length))
            # data['momentum'+'priceMA'+length] = talib.MOM(data['priceMA'+length],timeperiod=int(length))
            # plt.plot(-(data['priceMA'+length]-getzerotimevalue(data,openpricename+target[strategyname]['SWAP']))/1000,color='black')
            # plt.plot(data['momentum'+'priceMA'+length]/200-0.001,color='black',linestyle='--')
            # plt.fill_between(data.index,0.0001,0.002,where=data['momentum'+'priceMA'+length]<0.0,color='green',alpha=0.051)
            # plt.fill_between(data.index,0.0001,0.002,where=data['momentum'+'priceMA'+length]>0.0,color='red',alpha=0.051)
            # length = '60'
            # data['priceMA'+length] = talib.EMA(data[openpricename+target[strategyname]['SWAP']],int(length))
            # data['momentum'+'priceMA'+length] = talib.MOM(data['priceMA'+length],timeperiod=int(length))
            # plt.plot(-(data['priceMA'+length]-getzerotimevalue(data,openpricename+target[strategyname]['SWAP']))/1000,color='black')
            # plt.plot(data['momentum'+'priceMA'+length]/200-0.001,color='black',linestyle='--')
            # plt.fill_between(data.index,0.002,0.003,where=data['momentum'+'priceMA'+length]<0.0,color='green',alpha=0.051)
            # plt.fill_between(data.index,0.002,0.003,where=data['momentum'+'priceMA'+length]>0.0,color='red',alpha=0.051)

            # data['next'] = data[openpricename+target[strategyname]['SWAP']].diff(periods=4).shift(periods=-3,fill_value=0)#.fillna(0)
            # plt.fill_between(data.index,0.0001,0.002,where=data['next']<0.0,color='green',alpha=0.051)
            # plt.fill_between(data.index,0.0001,0.002,where=data['next']>0.0,color='red',alpha=0.051)





            # minvalue = data[openpricename+target[strategyname]['SWAP']].min()
            # maxvalue = data[openpricename+target[strategyname]['SWAP']].max()
            # data['normprice'] = (data[openpricename+target[strategyname]['SWAP']]-minvalue)/(maxvalue-minvalue)
            # plt.plot(-(data['normprice'])/500+0.002,color='black')


            data,sortinostrings = getstrategypositions(self.data_org,self.tlf,strategyname) # get strategy trade data
            figname = pairs+strategyname+'_'+datetimestring+'_'+str(self.args.env_seed)+sortinostrings
            plt.savefig(self.figfolder+'d'+figname+'.png',dpi=500, facecolor="azure", bbox_inches='tight', pad_inches=0)

            fig = self.tlf.figure('klines'+str(i)) # draw klines
            drawstrategycurves(self.data_org,self.tlf,self.instrument_ids,strategyname=strategyname)
            plt.savefig(self.figfolder+figname+'.png', dpi=500, facecolor="azure", bbox_inches='tight', pad_inches=0)
            #plt.tight_layout(pad=0)
            plt.show()
            plt.close('all')
        exit()






        # (valuesell-valuebuy)/valuebuy = e^(logvaluesell-logvaluebuy)-1 # e^x-1 = x+x^2/2+x^3/6
        splitratio= [8,2,0]
        traindata = data[                                                           :len(data)* splitratio[0]               //np.sum(splitratio)]
        validdata = data[len(data)* splitratio[0]               //np.sum(splitratio):len(data)*(splitratio[0]+splitratio[1])//np.sum(splitratio)]
        testdata  = data[len(data)*(splitratio[0]+splitratio[1])//np.sum(splitratio):]
        if len(testdata)==0: testdata = validdata
        print('traindata',len(traindata),'validdata',len(validdata),'testdata',len(testdata))
        self.traindata,self.validdata,self.testdata=traindata,validdata,testdata
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),])
            #torchvision.transforms.Normalize(mean=data_mean, std=data_std)])
        trainset = DatasetFromDataFrame(traindata,transform=transform)
        validset = DatasetFromDataFrame(validdata,transform=transform)
        testset  = DatasetFromDataFrame(testdata, transform=transform)
        #trainset = torch.utils.data.ConcatDataset([trainset0,validset])
        print('trainset:',len(trainset),'validset:',len(validset),'testset:',len(testset))
        self.trainset,self.validset,self.testset=trainset,validset,testset
        self.batch_size  = self.args.env_num
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.validloader = torch.utils.data.DataLoader(validset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False,num_workers=4)
        self.classes = ('up','none','down')
        self.g_step, self.record_interval = 0, 10
        foldername = self.args.exp_dir+'rewards/'
        os.makedirs(foldername,exist_ok=True)
        self.ftrain=open(foldername+'0_0','a')
        self.ftrain=open(foldername+'0_1','a')
        self.ftest =open(foldername+'0_2','a')
        self.testing, self.testingrewards = True, []
        self.pred,self.gold = [],[]
        obshape = self.reset().shape[1:]
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=obshape,dtype=np.uint8)
        self.action_space      = gym.spaces.Discrete(len(self.classes))
        self.reward_range      = [0,1]
        self.attr = {}
    def reset(self):
        ######################## for result check ########################################################
        if self.testing:
            data = pd.DataFrame(list(zip(self.pred,self.gold)),columns=['pred','gold'])
        else:
            data = pd.DataFrame(list(zip(self.pred,self.gold)),columns=['pred','gold'])
            data = pd.concat([self.testset.df.reset_index(drop=False,inplace=False), data], axis=1)
            #print(data)
            # print(data['pred'].value_counts())
            # print(data['gold'].value_counts())
            drawstrategycurves(self.data_org,self.tlf,self.instrument_ids,figname=str(self.args.env_seed),strategyname='arbitrage')
            plt.show()
            exit()
        self.pred,self.gold = [],[]
        ######################## for result check ########################################################
        if self.testing:
            self.dataiter = iter(self.testloader)
        else:
            print(int(self.g_step),',',int(np.mean(self.testingrewards)*10000),end='|',file=self.ftest,flush=True)
            print(int(self.g_step),',',int(np.mean(self.testingrewards)*10000))
            self.testingrewards = []
            self.dataiter = iter(self.trainloader)
        self.images, self.labels = self.dataiter.next()
        images = self.image_wrapper(self.images)
        return images
    def step(self, action):
        ######################## for result check ########################################################
        self.pred.extend(action)
        self.gold.extend(list(np.array(self.labels)))
        ######################## for result check ########################################################
        #print('step:',self.g_step,self.testing)
        reward = np.zeros(self.labels.shape[0])
        for i in range(len(reward)):
            if action[i]==self.labels[i]: reward[i] = 1
        if self.testing:
            self.testingrewards.append(reward)
            info = {'testing':True,'labels':[]}
        else:
            if self.g_step//self.batch_size%self.record_interval==0: # g_step increase batch_num each time
                print(int(self.g_step),',',int(np.mean(reward)*10000),end='|',file=self.ftrain,flush=True)
            self.g_step += self.batch_size
            info = {'testing':False,'labels':self.labels.numpy()}
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
        return images
    def render(self, mode='rgb_array', close=False):
        return None
    def close(self):
        print('',file=self.ftrain,flush=True)
        self.ftrain.close()
        print('',file=self.ftest,flush=True)
        self.ftest.close()
    def seed(self, seed=None):
        random.seed(seed)
########################################################################################################################
#unique,counts = np.unique(images,return_counts=True)
#print(np.asarray((unique,counts)).T)
#print(np.histogram(images, bins=[0, 1, 2, 3]))

# (valuesell-valuebuy)/valuebuy = e^(logvaluesell-logvaluebuy)-1 # e^x-1 = x+x^2/2+x^3/6
# if use log10, x,f(x) ; 0.3,1.0 ; 0.15,0.4 # <0.15,/3*8 ; >0.15,*4-0.2

########################################################################################################################
        # fig = plt.figure('func')
        # x = np.linspace(0,0.5,100)
        # plt.plot(x,np.power(np.e,x)-1,color='red')
        # # # x = np.linspace(0,0.15,150)
        # # plt.plot(x,x,color='black')

        # # plt.plot(x,(np.power(np.e,x)-1-x)/(np.power(np.e,x)-1),color='black')
        # # x = np.linspace(0.15,0.3,150)
        # # plt.plot(x,4*x-0.2,color='black')
        # plt.grid()
        # plt.show()
        # exit()



        # data['diff'] = data['openorg']-data['openorg'+'BINANCE_SWAP_ETH-USDT']
        # data['diff%']= data['diff']/data['openorg']#*100
        # fig = tlf.figure('diff%')
        # plt.plot(data['diff%'],color='red')

        # data['ratio']= data['difflog']-data['diff%']#np.log(data['diff%']+1)-data['difflog']#np.power(10,data['difflog'])-data['diff%']
        # fig = tlf.figure('ratio')
        # plt.plot(data['ratio'],color='green')

        # debug(data['ratio'])

        # profit = 0
        # for i in range(len(data)-1):
        #     profit += (data['openorg'][i+1]-data['openorg'][i])/data['openorg'][i]
        # print('profit',profit)
        # print(data['open'].values[-1]-data['open'][0])

        # fig=plt.figure('ex')
        # x = np.linspace(-1,1,100)
        # y = np.power(np.e,x)-1-x
        # plt.plot(x,y)
        # plt.grid()
        # print(y.min())



        # stdearn = data['open'].diff().fillna(0)
        # profits = data[prefix+'-total'].diff().fillna(0)
        # stdearn_sharpe = np.mean(stdearn)/np.std(stdearn)*np.sqrt(365)
        # profits_sharpe = np.mean(profits)/np.std(profits)*np.sqrt(365)
        # print('\n|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|',prefix,'\tprofits:',round(np.mean(profits),6),round(np.std(profits),6))
        # stdearn_sharpe = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=365)#525600)
        # profits_sharpe = sharpe_ratio(profits, risk_free=0, period='daily', annualization=365)#525600)
        # sortino_ratio_ = sortino_ratio(profits, required_return=0, period='daily', annualization=365)#525600)
        # print(  '|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|sortino_ratio:',sortino_ratio_)

        # stdearn = np.power(np.e,data['open'].diff().fillna(0))-1
        # profits = np.power(np.e,data[prefix+'-total'].diff().fillna(0))-1
        # stdearn_sharpe = np.mean(stdearn)/np.std(stdearn)*np.sqrt(365)
        # profits_sharpe = np.mean(profits)/np.std(profits)*np.sqrt(365)
        # print('\n|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|',prefix,'\tprofits:',round(np.mean(profits),6),round(np.std(profits),6))
        # stdearn_sharpe = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=365)#525600)
        # profits_sharpe = sharpe_ratio(profits, risk_free=0, period='daily', annualization=365)#525600)
        # sortino_ratio_ = sortino_ratio(profits, required_return=0, period='daily', annualization=365)#525600)
        # print(  '|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|sortino_ratio:',sortino_ratio_)

        # stdearn = data['openorg'].diff().shift(periods=-1,fill_value=0).fillna(0)/data['openorg']
        # tmp     = np.power(np.e,data[prefix+'-total'])
        # profits = tmp.diff().shift(periods=-1,fill_value=0).fillna(1)/tmp
        # debug(tmp,30)
        # debug(profits,30)
        # stdearn_sharpe = np.mean(stdearn)/np.std(stdearn)*np.sqrt(365)
        # profits_sharpe = np.mean(profits)/np.std(profits)*np.sqrt(365)
        # print('\n|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|',prefix,'\tprofits:',round(np.mean(profits),6),round(np.std(profits),6))
        # stdearn_sharpe = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=365)#525600)
        # profits_sharpe = sharpe_ratio(profits, risk_free=0, period='daily', annualization=365)#525600)
        # sortino_ratio_ = sortino_ratio(profits, required_return=0, period='daily', annualization=365)#525600)
        # print(  '|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|sortino_ratio:',sortino_ratio_)


            # indicator   = 'EMA'#'SMA'
            # lengths1    = [7]#[480]#[1800]#[500]
            # lengths2    = [3]#[30]#[11]#,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]#[200]
            # parameters  = [30]#[1440]#[3600]#[1200]
            # period      = 50
            # getsignals(data,indexies=lengths1,indexjes=lengths2,indicator=indicator,parameters=parameters)
            # #drawtraces(data,indexies=lengths1,indexjes=lengths2,savefig=True)
            # drawcurves(data,indexies=lengths1,indexjes=lengths2,indicator=indicator,parameters=parameters,period=period,showfig=True,savefig=True,figsize=(64,18),foldername=self.args.exp_dir)
            # #getprofits(data,indexies=lengths1,indexjes=lengths2,savefig=True)
            # #getheatmap(data,indexies=[i for i in range(0,len(data),period)],lengths1=lengths1,lengths2=lengths2,period=period,savefig=True,figsize=(16,16),maxrange=20)
            # exit()
            #drawstrategycurves(self.data_org,self.tlf,self.instrument_ids,figname=str(self.args.env_seed),strategyname='random')
            # fig = self.tlf.figure('amount')
            # plt.plot(data['amount'],color='black')
            # fig = self.tlf.figure('count')
            # plt.plot(data['count'],color='black')

        # count_mean1 = data['count'].rolling(5).mean()
        # count_mean2 = talib.SMA(data['count'],5)
        # debug(data['count'],20)
        # debug(count_mean1,20)
        # debug(count_mean2,20)
        # exit()

        # data['upper'],data['middle'],data['lower'] = talib.BBANDS((data['open']-data['open'][0])/1000)#,matype=talib.MA_Type.EMA,timeperiod=10,nbdevup=2,nbdevdn=2)
        # debug(data,30)
        # exit()


