import talib,influxdb,functools,os,tqdm,pprint,time,datetime,imageio,cv2,random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
COLORS = ['red','orange','green','cyan','blue','purple']#,'black']
MARKERS= ['+', '.', 'o', '*']#",",".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","d","D","|","_","","","","",""
import yfinance as yf
import mplfinance as mpf
def timer(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        print(func.__name__,"start...")
        tic = time.time()
        result = func(*args, **kwargs)
        print(func.__name__,"done. Used",time.time()-tic,"s")
        print('')
        return result
    return wrapped_func
def folder(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        folder = './'+func.__name__+'/'
        os.makedirs(folder,exist_ok=True)
        result = func(*args, **kwargs, folder=folder)
        return result
    return wrapped_func
def looper(func):
    @functools.wraps(func)
    def wrapped_func(*args, indexies, **kwargs):
        results = []
        for indexi in tqdm.tqdm(indexies):
            results.append(func(*args, **kwargs, indexi=indexi))
        return results
    return wrapped_func
def looper2(func):
    @functools.wraps(func)
    def wrapped_func(*args, indexies,indexjes, **kwargs):
        resultmatrix = []
        for indexi in tqdm.tqdm(indexies):
            resultvector = []
            for indexj in tqdm.tqdm(indexjes):
                resultvector.append(func(*args, **kwargs, indexi=indexi, indexj=indexj))
            resultmatrix.append(resultvector)
        return resultmatrix
    return wrapped_func
def plotter(func):
    @functools.wraps(func)
    def wrapped_func(*args, showfig=False,savefig=False,figsize=(32,9),foldername='./',surfix='', **kwargs):
        folder = foldername+func.__name__+'/'
        os.makedirs(folder,exist_ok=True)
        fig = plt.figure(figsize=figsize)
        result, figname = func(*args, **kwargs)
        if savefig and figname!=None: plt.savefig(folder+figname+surfix+'.png', dpi=200, facecolor="azure", bbox_inches='tight', pad_inches=0)
        if showfig and figname!=None: plt.show()
        plt.close()
        return result
    return wrapped_func


def debug(data,num=100):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print('data.dtypes',data.dtypes)#print(data[data['15-60-sellsignal']==True].head(15))
    if num>0: print(data.head(num))
    else: print(data.tail(-num))
    print('len(data):',len(data))
def indicators(data,indicator,indexes):
    for index in indexes:
        if indicator=='EMA': data[indicator+str(index)]=talib.EMA((data['open']+data['close'])/2,index).fillna(0)
        if indicator=='SMA': data[indicator+str(index)]=talib.SMA((data['open']+data['close'])/2,index).fillna(0)
    return data
    # fill 0 to first x items, not sure if it is correct way to handle it
    # need interpolate such as data.interpolate(method='pad', limit=2) before, otherwise, all data after NaN become NaN
def logdata(data):
    data['openorg'] = data['open']
    data['closeorg']= data['close']
    data['highorg'] = data['high']
    data['loworg']  = data['low']
    data['open'] = np.log(data['openorg'])
    data['close']= np.log(data['closeorg'])
    data['high'] = np.log(data['highorg'])
    data['low']  = np.log(data['loworg'])
    return data
@timer
def groupdata(data_org,length=1):
    if length!=1:
        nums = (len(data_org)-1)//length+1
        records = []
        for i in tqdm.tqdm(range(nums)):
            slicei = data_org[i*length:(i+1)*length]
            record={'time':  slicei['time'].values[0],
                    'open':  slicei['open'].values[0],
                    'close': slicei['close'].values[-1],
                    'high':  slicei['high'].max(),
                    'low':   slicei['low'].min(),
                    'volume':slicei['volume'].sum(),
                    'count': slicei['count'].sum(),
                    'amount':slicei['amount'].sum(),
                    'volume_long':slicei['volume_long'].sum(),
                    'count_long': slicei['count_long'].sum(),
                    'amount_long':slicei['amount_long'].sum(),
                    'volume_base':slicei['volume_base'].sum()}
            # for ER(efficient ratio)
            record['ER'] = np.abs(record['close']-record['open'])/(slicei['close']-slicei['open']).abs().sum()
            record['PD'] = (record['high']-record['low'])/(slicei['high']-slicei['low']).sum()
            record['VT'] = ((slicei['close']-slicei['open'])/slicei['open']).std()
            records.append(record)
        data = pd.DataFrame(records)
    else: data = data_org
    data['time'] = pd.to_datetime(data['time'],utc=True)
    data.info()# print('Max one line difference:',(dfs['high']-dfs['low']).max())
    return data
@timer
def expanddata(data,datetime0,datetime1,freq):
    idx  = pd.date_range(datetime0, datetime1, freq=freq)
    idx  = idx.tz_localize('UTC')#tz_convert('UTC')#to_pydatetime()
    data_filled = pd.DataFrame({'time':idx})
    data_filled = data_filled.merge(data,how='left',on='time')#outer
    expand_ratio= (len(data_filled)-1)/(len(data)-1)
    assert expand_ratio==(len(data_filled)-1)//(len(data)-1) and expand_ratio!=0
    for column in data_filled.columns:
        if column=='time': continue
        data_filled[column] = data_filled[column].interpolate()
        if column=='volume' or column=='count' or column=='amount' \
            or column=='volume_long' or column=='count_long' or column=='amount_long' \
            or column=='volume_base': # only approximately right... 
            data_filled[column] = data_filled[column]/expand_ratio
    data_filled.info()
    return data_filled
@timer
def filldata(data,datetime0,datetime1):
    idx  = pd.date_range(datetime0, datetime1, freq='1min')
    idx  = idx.tz_localize('UTC')#tz_convert('UTC')#to_pydatetime()
    data_filled = pd.DataFrame({'time':idx})
    data_filled = data_filled.merge(data,how='left',on='time')#outer
    for column in data_filled.columns:
        if column=='time': continue
        data_filled[column] = data_filled[column].interpolate() # linear interpolate missing data
        data_filled[column] = data_filled[column].interpolate(method='bfill') # fill all past NaN by first unNaN value
    data_filled.info()
    return data_filled
@timer
def getdata(host,port,querycommand):
    client = influxdb.InfluxDBClient(host=host, port=port)
    points = client.query(querycommand).get_points()
    data = pd.DataFrame([point for point in points])
    data['time'] = pd.to_datetime(data['time'],utc=True)
    data.info()
    return data
def getdatetimestring(timestring):
    return str(int(datetime.datetime.strptime(timestring,'%Y-%m-%d %H:%M:%S %z').timestamp()*1000))+'ms'#,%f %z').timestamp()*1000))+'ms'


@looper2
@plotter
def getprofits(data,indexi,indexj):
    prefix = str(indexj)+'-'+str(indexi)
    if prefix not in data.columns: return 0,None
    buytable  = pd.DataFrame(data[data[prefix+'-buyprice'+'-long']!=0][prefix+'-buyprice'+'-long']).reset_index()#drop=True, inplace=True)
    selltable = pd.DataFrame(data[data[prefix+'-sellprice'+'-long']!=0][prefix+'-sellprice'+'-long']).reset_index()#drop=True, inplace=True)
    if len(buytable)>len(selltable): buytable = buytable[:-1]
    tradetable= pd.DataFrame({'buyindex':buytable['index'],  'buyprice':buytable[prefix+'-buyprice'+'-long'],
                              'sellindex':selltable['index'],'sellprice':selltable[prefix+'-sellprice'+'-long']})
    tradetable['profit'] = -tradetable['sellprice']/tradetable['buyprice']-1
    #print(tradetable['profit'].describe(include='all'))
    ax = tradetable['profit'].hist(histtype='bar', bins=100)#, normed=True)
    ax.set_xlim((-0.1,0.1))
    profit = tradetable['profit'].sum()*525600/len(data)
    return profit,prefix
@timer
@looper
@plotter
def getheatmap(data,indexi,lengths1,lengths2,period,maxrange=None):
    profitmatrix = getprofits(data[indexi:indexi+period],indexies=lengths1,indexjes=lengths2)
    profitmatrix = pd.DataFrame(profitmatrix,index=lengths1,columns=lengths2)
    if maxrange==None: maxrange = profitmatrix.abs().max().max()
    ax = sns.heatmap(profitmatrix, annot=True, fmt='.1f', cmap='coolwarm', vmin=-maxrange, vmax=maxrange)
    return None,str(indexi)
@timer
@looper2
@plotter
def drawtraces(data,indexi,indexj):
    prefix = str(indexj)+'-'+str(indexi)
    if prefix not in data.columns: return None,None
    for index, row in data.iterrows():
        if row[prefix+'-buyprice'+'-long']==0: continue
        zeroprice = row[prefix+'-buyprice'+'-long']
        dataslice = data[index-indexj:index+indexi][['low','high','open','close']]
        dataslice = (dataslice-zeroprice).reset_index()
        plt.fill_between(dataslice.index, dataslice['low'], dataslice['high'], facecolor='grey', alpha=0.75)
        plt.fill_between(dataslice.index, dataslice['open'],dataslice['close'],facecolor='black',alpha=0.75)
    return None,prefix
@timer
@looper
@plotter
def drawcurves(data,indexi,indexjes,indicator,parameters,period,linewidth=.21):
    shiftvalue = data['ocmid'][data.index[0]]
    figname = indicator+str(indexi)+'x'+'-'.join([str(indexj) for indexj in indexjes])
    colors = cycle(COLORS)
    for indexj in tqdm.tqdm(indexjes):
        prefix = str(indexj)+'-'+str(indexi) # if short, order should be inverse
        if prefix not in data.columns: continue
        color=next(colors)
        plt.plot(data[indicator+str(indexj)],color=color,alpha=1.,linewidth=linewidth)
        plt.scatter(data[data[prefix+'-buyprice'+'-long']!=0].index, data[data[prefix+'-buyprice'+'-long']!=0]['open'], color=color, marker='+', s=35)
        plt.scatter(data[data[prefix+'-sellprice'+'-long']!=0].index,data[data[prefix+'-sellprice'+'-long']!=0]['open'],color=color, marker='.', s=35)
        shiftvalueS = data['ocmid'][min(data[data[prefix+'-buyprice'+'-long']!=0].index,default=[data.index[0]])]###
        plt.plot(data[prefix+'-total']+shiftvalueS,color=color,alpha=1.,linewidth=linewidth)
    plt.plot(data[indicator+str(indexi)],       color='black',alpha=1.,linewidth=linewidth)
    plt.plot(data[indicator+str(parameters[0])],color='black',alpha=1.,linewidth=linewidth,linestyle='--')###
    plt.plot(data['ocmid'],color='grey',alpha=1.,linewidth=linewidth)
    #plt.fill_between(data.index, data['low'], data['high'], facecolor='grey', alpha=0.5)
    #plt.fill_between(data.index, data['open'],data['close'],facecolor='black',alpha=0.5)
    plt.fill_between(data.index, data['count']*0+shiftvalue, data['count']/50+shiftvalue, facecolor='grey', alpha=0.375)
    plt.gca().set_xticks([i+data.index[0] for i in range(0,len(data),period)])
    plt.gca().set_xticklabels([data['time'][data.index[i]] for i in range(0,len(data),period)])
    #plt.gca().set_yticks([i for i in range(int(plt.ylim()[0]),int(plt.ylim()[1]),1000)])
    plt.grid(color='black',linewidth=.1)
    plt.xticks(rotation=90,Fontsize=10)
    return None,figname


def update_state(state,signal): # if first signal=-1, first state will equal it and =-1 too and get assert error
    assert state  in (0,1),    "state={}".format(state)
    assert signal in (0,1,-1), "signal={}".format(signal)
    if signal == 1 or state == 1 and signal!=-1: return 1
    else: return 0
update_state_numpy = np.frompyfunc(update_state,2,1)
update_state_accum = lambda x: update_state_numpy.accumulate(x,dtype=object).astype(int)
def getpositions(data,tradesignal,strategyname,direction,target,openpricename,closepricename,holdpricename,fee=0.0,suffix=''):
    prefix = strategyname+direction+target
    if tradesignal.iloc[0]==-1: tradesignal.iloc[0]=0 # solution: if first signal=-1, let it =0
    tradesignal[data[openpricename+target].isnull()]=0 # if there is signal on price=np.nan place, flow,rest and total calculation will wrong
    tradesignal[data[closepricename+target].isnull()]=0
    data[prefix+'-holdperiod'+suffix]  = update_state_accum(tradesignal)
    data[prefix+'-tradetiming'+suffix] = data[prefix+'-holdperiod'+suffix].diff().fillna(data[prefix+'-holdperiod'+suffix]).astype(int)
    data[prefix+'-opentiming'+suffix]  = data[prefix+'-tradetiming'+suffix].clip(lower=0)
    data[prefix+'-closetiming'+suffix] =-data[prefix+'-tradetiming'+suffix].clip(upper=0)
    data[prefix+'-openprice'+suffix]   = data[prefix+'-opentiming'+suffix] *(data[openpricename+target] +fee)#low,high
    data[prefix+'-closeprice'+suffix]  = data[prefix+'-closetiming'+suffix]*(data[closepricename+target]-fee)
    data[prefix+'-hold'+suffix] = data[prefix+'-holdperiod'+suffix]*(data[holdpricename+target])# capital curves
    data[prefix+'-flow'+suffix] =-data[prefix+'-openprice'+suffix]+data[prefix+'-closeprice'+suffix]
    data[prefix+'-rest'+suffix] = data[prefix+'-flow'+suffix].cumsum()
    data[prefix+'-total'+suffix]= data[prefix+'-hold'+suffix]+data[prefix+'-rest'+suffix]
    return data
#def drawtrademark():
# def getpositions(data,prefix,tradetime,suffix):
#     data[prefix+'-holdtime2'+suffix]   = update_state_accum(tradetime)
#     data[prefix+'-tradetime2'+suffix]  = data[prefix+'-holdtime2'+suffix].diff()
#     data.loc[data.index[0],prefix+'-tradetime2'+suffix] = 0
#     data[prefix+'-buytime2'+suffix]    = data[prefix+'-tradetime2'+suffix].clip(lower=0)
#     data[prefix+'-selltime2'+suffix]   = data[prefix+'-tradetime2'+suffix].clip(upper=0)
#     data[prefix+'-buyprice'+suffix]    = data[prefix+'-buytime2'+suffix] *data['open']#*1.001#low,high
#     data[prefix+'-sellprice'+suffix]   = data[prefix+'-selltime2'+suffix]*data['open']#*0.009
#     data[prefix+'-hold'+suffix] = data[prefix+'-holdtime2'+suffix]*data['ocmid']# capital curves
#     data[prefix+'-flow'+suffix] =-data[prefix+'-buyprice'+suffix]-data[prefix+'-sellprice'+suffix]
#     data[prefix+'-rest'+suffix] = data[prefix+'-flow'+suffix].cumsum()
#     data[prefix+'-total'+suffix]= data[prefix+'-hold'+suffix]+data[prefix+'-rest'+suffix]
#     #debug(data,-100)
#     #exit()
from empyrical import annual_return,sharpe_ratio,sortino_ratio
def calsharpes(benchmarks,strategies,annualization=365):#525600)
    benchmarks_sharpe = np.mean(benchmarks)/np.std(benchmarks)*np.sqrt(annualization)
    strategies_sharpe = np.mean(strategies)/np.std(strategies)*np.sqrt(annualization)
    print('\n|benchmarks:',str(format(benchmarks_sharpe,'+.6f')),'|strategies:',str(format(strategies_sharpe,'+.6f'))
        ,'|mean,std:',str(format(np.mean(strategies),'+.6f')),str(format(np.std(strategies),'+.6f')))
    benchmarks_sharpe = sharpe_ratio(benchmarks, period='daily', annualization=annualization, risk_free=0)
    strategies_sharpe = sharpe_ratio(strategies, period='daily', annualization=annualization, risk_free=0)
    print(  '|benchmarks:',str(format(benchmarks_sharpe,'+.6f')),'|strategies:',str(format(strategies_sharpe,'+.6f')))
    benchmarks_sortino=sortino_ratio(benchmarks, period='daily', annualization=annualization, required_return=0)
    strategies_sortino=sortino_ratio(strategies, period='daily', annualization=annualization, required_return=0)
    print(  '|benchmarks:',str(format(benchmarks_sortino,'+.6f')),'|strategies:',str(format(strategies_sortino,'+.6f')))
    return benchmarks_sortino,strategies_sortino
def getsignal(data,indexi,indexj,indicator,parameters,threshold):
    prefix = str(indexj)+'-'+str(indexi)
    data[prefix+'-period']   =(data[prefix]>threshold)
    data[prefix+'-2'] = data[indicator+str(indexi)]-data[indicator+str(parameters[0])]
    data[prefix+'-period'+'-2']   =(data[prefix+'-2']>threshold)
    data[prefix+'-3'] = data['low']-data[indicator+str(indexi)]
    data[prefix+'-period'+'-3']   =(data[prefix+'-3']>threshold)
    data[prefix+'-period'] = data[prefix+'-period']&data[prefix+'-period'+'-2']&data[prefix+'-period'+'-3']
    data[prefix+'-period'] = data[prefix+'-period'].shift(periods=1,fill_value=False)
    if 'preds' in data.columns: data[prefix+'-period'] = (data['preds']>0)
    # series = [random.gauss(0.0, 1.0) for i in range(len(data))]
    # series = pd.Series(series)
    # data[prefix+'-period'] = (series>0)
    data[prefix+'-switch'] = data[prefix+'-period'].astype(int).diff()
    data.loc[data.index[0]                   ,prefix+'-switch'] = 0
    data.loc[data.index[(indexi-1)%len(data)],prefix+'-switch'] = 0 # if indexi > len(data), this line's modify make data['time'] can not correctly draw
    data[prefix+'-tradetime'] = data[prefix+'-switch']
    #if 'preds' not in data.columns: data[prefix+'-tradetime'] = data[prefix+'-tradetime'].shift(periods=1,fill_value=0)
@timer
@looper2
def getsignals(data,indexi,indexj,indicator,parameters):
    if indexj>=indexi:# or indexj<indexi//105:
        return
    prefix = str(indexj)+'-'+str(indexi)
    data[prefix]=data[indicator+str(indexj)]-data[indicator+str(indexi)]
    getsignal(data,indexi,indexj,indicator,parameters,threshold=-100.1)
    getpositions(data,prefix=prefix,tradetime= data[prefix+'-tradetime'],suffix='-long')
    getsignal(data,indexi,indexj,indicator,parameters,threshold= 100.1)
    getpositions(data,prefix=prefix,tradetime=-data[prefix+'-tradetime'],suffix='-short')
    data[prefix+'-total'] = data[prefix+'-total'+'-long']-data[prefix+'-total'+'-short']
    stdearn = data['open'].diff().fillna(0)
    profits = data[prefix+'-total'].diff().fillna(0)
    benchmarks_sortino,strategies_sortino=calsharpes(stdearn,profits)
    return


from statsmodels.tsa.arima_model import ARMA
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX ###
from pandas.plotting import autocorrelation_plot   
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.stattools.arma_order_select_ic',FutureWarning)
warnings.filterwarnings('ignore')
def drawfunc(datas, func, ylim):
    fig = plt.figure(func.__name__,figsize=(8,8))
    colors = cycle(COLORS)
    for data in datas:
        datafunc = func(data)
        print(datafunc)
        color = next(colors)
        plt.plot(datafunc,color=color,marker='o')#,linewidth=.21)
    plt.ylim(ylim)
    plt.grid()
    n   = len(datas[0])
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    plt.axhline(y= z99/np.sqrt(n), linestyle='--', color='grey')
    plt.axhline(y= z95/np.sqrt(n), color='grey')
    plt.axhline(y= 0.0, color='black')
    plt.axhline(y=-z95/np.sqrt(n), color='grey')
    plt.axhline(y=-z99/np.sqrt(n), linestyle='--', color='grey')
def stctests(datas):
    for data in datas:
        adf = st.adfuller(data)
        print('adf:',adf)
        kps = st.kpss(data)
        print('kps:',kps)
        wht = acorr_ljungbox(data, lags=list(range(1,10,1)))#, boxpierce=True)
        print('wht:',wht)
def drawdata(datas,name,figsize=(16,16)):
    fig = plt.figure(name,figsize=figsize)
    colors = cycle(COLORS)
    for data in datas:
        color = next(colors)
        plt.plot(data,color=color,linewidth=.21,marker='.')
    plt.grid()
def select_order(data,max_ar,max_ma):
    order = st.arma_order_select_ic(data,max_ar=max_ar,max_ma=max_ma,ic=['aic', 'bic', 'hqic'])
    print(order)
def datarecorder(data):
    f = open('data.txt','w')
    for datai in data:
        print(datai,file=f,flush=True)
    f.close()
def datareader():
    lines = open('data.txt','r').readlines()
    data = []
    for line in lines:
        data.append(float(line))
    return pd.Series(data)
def statics(data):
    return data
def main():
    measurements = 'bfs.autogen.klinehistory'
    instrument_id= 'BINANCE_SPOT_ETH-USDT'
    tags = 'open,close,high,low,amount,amount_long,count'#,count_long,volumn,volumn_long'#amount,amount_long,count,count_long,volumn,volumn_base,volumn_long
    headdatetime = getdatetimestring('20.12.2021 00:00:00,00')
    taildatetime = getdatetimestring('30.12.2021 00:00:00,00')
    querycommand = 'select '+tags+' from '+measurements+' where '+'time<'+taildatetime+' and '+'time>'+headdatetime+' and '+"instrument_id='"+instrument_id+"'"
    data = getdata(host='47.57.72.27', port=8086, querycommand=querycommand)
    data = groupdata(data,length=1440)
    data = statics(data)
    indicator   = 'EMA'#'SMA'
    lengths1    = [7]#[480]#[1800]#[500]
    lengths2    = [3]#[30]#[11]#,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]#[200]
    parameters  = [30]#[1440]#[3600]#[1200]
    period      = 50
    data = indicators(data,indicator=indicator,indexes=sorted(list(dict.fromkeys(lengths1+lengths2+parameters))))
    getsignals(data,indexies=lengths1,indexjes=lengths2,indicator=indicator,parameters=parameters)
    #drawtraces(data,indexies=lengths1,indexjes=lengths2,savefig=True)
    drawcurves(data,indexies=lengths1,indexjes=lengths2,indicator=indicator,parameters=parameters,period=period,showfig=True,savefig=True,figsize=(64,18))
    #getprofits(data,indexies=lengths1,indexjes=lengths2,savefig=True)
    #getheatmap(data,indexies=[i for i in range(0,len(data),period)],lengths1=lengths1,lengths2=lengths2,period=period,savefig=True,figsize=(16,16),maxrange=20)
if __name__ == '__main__':
    main()
#lengths1 = [10080,43200]#[120]#60,90,120,240,360,720,1440,2880,5760,10080]
#lengths2= [1440,2880,5760,10080]#[20,30,45,60,90,120]
#lengths1 = [15]#60,120,240]#,420,1440]#[5,15,30,60,120,240,1440]#7,25,100;5,10,20,30,60,120#list(range(200,2000,200))#[27,81,243,729,2187]
#lengths2 = [5]#,15,30]#,60,120]#[5,15,30,60,120,240,1440]#10080,43200#list(range(50,200,50))#[9,27,81,243,729,2187]
#lengths1 = list(np.logspace(6,13,9,base=2).astype(int))
#lengths2= list(np.logspace(2,10,6,base=2).astype(int))
    # data = yf.download('ETH-USD', start='2021-12-28', end='2021-12-29', interval='1m')#period='7d' 'SPY AAPL' EURUSD=X ETH-USD
    # #mpf.plot(data,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)
    # data.index = pd.DatetimeIndex(data.index)
    # idx = pd.date_range('2021-12-27 15:00:00+00:00', '2021-12-28 15:00:00+00:00', freq='1min')
    # data = data.reindex(idx, method='ffill')#fill_value=0)
    # data = data.reset_index()
    # data = data.rename(columns={'index':'time','Open':'open','Close':'close','High':'high','Low':'low','Volume':'count','Adj Close':'close-adj'})
    # data['count'] = data['count']/100000

# def statics(data):
#     return data
#     #debug(data,20)
#     data['ocdiff%'] = data['ocdiff']/data['open']*100
#     #data = data[1000:3000]
#     return data
#     delta = data['ocdiff%']
#     # stctests([delta])
#     # drawfunc([delta], st.acf,[-0.15,0.15])
#     # drawfunc([delta],st.pacf,[-0.15,0.15])
#     #plt.show()
#     #exit()

#     model = arch_model(delta, mean='Zero', vol='GARCH', p=1, q=1)
#     model_fit = model.fit()
#     print(model_fit.summary)
#     # #yhat = model_fit.forecast(horizon=1)
#     # preds = model_fit.forecast()#align='target')
#     # preds = preds.volatility
#     # print(preds)
#     # #exit()
#     drawdata([delta,model_fit.conditional_volatility,-model_fit.conditional_volatility],'preds',figsize=(16,16))
#     #fig = model_fit.hedgehog_plot()
#     #fig = model_fit.plot()

#     plt.show()
#     exit()
#     if 1:
#         size = int(len(delta)*0.996)
#         train, tests = delta[0:size], delta[size:len(delta)]
#         # order = st.arma_order_select_ic(train,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
#         # print(order)
#         # exit()
#         model = ARMA(train, order=(6,5))
#         model_fit = model.fit(disp=-1, method='css')
#         preds = model_fit.predict()
#         resid = pd.Series(model_fit.resid)
#         print(delta)
#         print(train)
#         print(resid)
#         print(tests)
#         fores = model_fit.forecast(len(tests))
#         print(fores)
#         fores = pd.Series(fores[0],index=tests.index)
#         print(fores)
#         error = tests-fores
#         print(error)
#         stctests([delta,preds,resid,fores,error])
#         #drawfunc([delta,preds,resid,fores,error], st.acf,[-0.15,0.15])
#         #drawfunc([delta,preds,resid,fores,error],st.pacf,[-0.15,0.15])
#         drawdata([delta,preds,resid,fores,error],'preds',figsize=(16,16))

#         plt.show()
#         exit()

#         history = [x for x in train]
#         predictions0 = list()
#         predictions1 = list()
#         predictions2 = list()
#         # walk-forward validation
#         for t in tqdm.tqdm(range(len(test))):
#             model = ARMA(history, order=(0,1))
#             #model = ARIMA(history, order=(5,1,0))
#             model_fit = model.fit(disp=-1)
#             output = model_fit.forecast(10)
#             predictions0.append(output[0][0])
#             predictions1.append(output[0][4])
#             predictions2.append(output[0][9])
#             history.append(test[t])
#             #print('predicted=%f, expected=%f' % (output[0], test[t]))
#         # evaluate forecasts
#         rmse0 = np.sqrt(mean_squared_error(test, predictions0))
#         print('Test RMSE0: %.3f' % rmse0)
#         rmse1 = np.sqrt(mean_squared_error(test, predictions1))
#         print('Test RMSE1: %.3f' % rmse1)
#         rmse2 = np.sqrt(mean_squared_error(test, predictions2))
#         print('Test RMSE2: %.3f' % rmse2)
#         # plot forecasts against actual outcomes
#         plt.plot(test)
#         plt.plot(predictions0, color='red')
#         plt.plot(predictions1, color='orange')
#         plt.plot(predictions2, color='green')
#         plt.show()
#         exit()

#     infolength = 200
#     preds = [0]*infolength
#     resid = [0]*infolength
#     for t in tqdm.tqdm(range(len(delta)-infolength)):
#         model = ARIMA(delta[t:t+infolength], order=(3,0,3))
#         model_fit = model.fit()
#         resid.append(model_fit.resid.mean())
#         output = model_fit.forecast(1)
#         preds.append(output.values[0])
#     preds = pd.Series(preds,index=delta.index)
#     resid = pd.Series(resid,index=delta.index)
#     error = delta-preds
#     stctests([delta,preds,error,resid])
#     drawfunc([delta,preds,error,resid], st.acf,[-0.15,0.15])
#     drawfunc([delta,preds,error,resid],st.pacf,[-0.15,0.15])
#     drawdata([delta,preds,error,resid],'preds',figsize=(16,16))
#     data['preds'] = preds
#     return data
#     # rs = []
#     # for inum in range(100):
#     #     series = [random.gauss(0.0, 1.0) for i in range(300*(inum+1))]
#     #     series = pd.Series(series)
#     #     r = acorr_ljungbox(series, lags=[1])
#     #     print(r)
#     # exit()

    # profits = data['ocdiff%']*(data[prefix+'-period'].astype(int)*2-1)
    # profits_sharpe = round(np.mean(profits)/np.std(profits)*np.sqrt(525600),6)
    # stdearn = data['ocdiff%']
    # stdearn_sharpe = round(np.mean(stdearn)/np.std(stdearn)*np.sqrt(525600),6)
    # # print(profits)
    # # print(stdearn)
    # # print(data[prefix+'-period'])
    # # exit()
    # print('\n|',stdearn_sharpe,profits_sharpe,'\t|',prefix,'profits:\t',round(np.mean(profits),6),round(np.std(profits),6))
    # sp_ratio = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=525600)
    # print(sp_ratio)
    # sp_ratio = sharpe_ratio(profits, risk_free=0, period='daily', annualization=525600)
    # print(sp_ratio)
    # st_ratio = sortino_ratio(profits, required_return=0, period='daily', annualization=525600)
    # print(st_ratio)





