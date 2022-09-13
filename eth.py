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
    def wrapped_func(*args, showfig=False,savefig=False,figsize=(32,9), **kwargs):
        folder = './'+func.__name__+'/'
        os.makedirs(folder,exist_ok=True)
        fig = plt.figure(figsize=figsize)
        result, figname = func(*args, **kwargs)
        if savefig and figname!=None: plt.savefig(folder+figname+'.png', dpi=200, facecolor="azure", bbox_inches='tight', pad_inches=0)
        if showfig and figname!=None: plt.show()
        plt.close()
        return result
    return wrapped_func
def debug(data,num=100):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(data.dtypes)
    #print(data[data['15-60-sellsignal']==True].head(15))
    if num>0: print(data.head(num))
    else: print(data.tail(-num))
    print(len(data))
    return
@timer
def groupdata(data,length=1):
    if length!=1:
        nums = len(data)//length
        records = []
        for i in tqdm.tqdm(range(nums)):
            slicei = data[i*length:(i+1)*length]
            record = {'time':slicei['time'].values[0],'open':slicei['open'].values[0],'close':slicei['close'].values[-1],
                        'high':slicei['high'].max(),'low':slicei['low'].min(),'count':slicei['count'].mean()}
            records.append(record)
        dfs = pd.DataFrame(records)#,['open','close','high','low'])
    else: dfs = data
    dfs.info()
    dfs['ocdiff']=  dfs['close']-dfs['open']
    dfs['hldiff']=  dfs['high'] -dfs['low']
    print('Max one line difference:',dfs['hldiff'].max())
    dfs['ocmid'] = (dfs['close']+dfs['open'])/2
    dfs['hlmid'] = (dfs['high'] +dfs['low'])/2
    return dfs
@timer
def getdata(host,port,querycommand):
    client = influxdb.InfluxDBClient(host=host, port=port)
    result = client.query(querycommand)
    points = result.get_points()
    data = []
    for point in points:
        data.append(point)
    data = pd.DataFrame(data)
    data.info()
    return data
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


def update_state(state,signal):
    assert state  in (0,1),    "state={}".format(state)
    assert signal in (0,1,-1), "signal={}".format(signal)
    if signal == 1 or state == 1 and signal!=-1: return 1
    else: return 0
update_state_numpy = np.frompyfunc(update_state,2,1)
update_state_accum = lambda x: update_state_numpy.accumulate(x,dtype=object).astype(int)
def getpositions(data,prefix,tradetime,suffix):
    data[prefix+'-holdtime2'+suffix]   = update_state_accum(tradetime)
    data[prefix+'-tradetime2'+suffix]  = data[prefix+'-holdtime2'+suffix].diff()
    data.loc[data.index[0],prefix+'-tradetime2'+suffix] = 0
    data[prefix+'-buytime2'+suffix]    = data[prefix+'-tradetime2'+suffix].clip(lower=0)
    data[prefix+'-selltime2'+suffix]   = data[prefix+'-tradetime2'+suffix].clip(upper=0)
    data[prefix+'-buyprice'+suffix]    = data[prefix+'-buytime2'+suffix] *data['open']#*1.001#low,high
    data[prefix+'-sellprice'+suffix]   = data[prefix+'-selltime2'+suffix]*data['open']#*0.009
    data[prefix+'-hold'+suffix] = data[prefix+'-holdtime2'+suffix]*data['ocmid']# capital curves
    data[prefix+'-flow'+suffix] =-data[prefix+'-buyprice'+suffix]-data[prefix+'-sellprice'+suffix]
    data[prefix+'-rest'+suffix] = data[prefix+'-flow'+suffix].cumsum()
    data[prefix+'-total'+suffix]= data[prefix+'-hold'+suffix]+data[prefix+'-rest'+suffix]
    #debug(data,-100)
    #exit()
@looper
def indicators(data,indexi,indicator):
    if indicator=='EMA': data[indicator+str(indexi)]=talib.EMA(data['ocmid'],indexi)
    if indicator=='SMA': data[indicator+str(indexi)]=talib.SMA(data['ocmid'],indexi)
from empyrical import annual_return,sharpe_ratio,sortino_ratio
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
    #debug(data, 30)
    #debug(data,-30)
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

    stdearn = data['ocdiff%']
    profits = data['ocdiff%']*(data[prefix+'-period'].astype(int)*2-1)
    stdearn_sharpe = np.mean(stdearn)/np.std(stdearn)*np.sqrt(365)
    profits_sharpe = np.mean(profits)/np.std(profits)*np.sqrt(365)
    print('\n|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|',prefix,'\tprofits:',round(np.mean(profits),6),round(np.std(profits),6))
    stdearn_sharpe = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=365)#525600)
    profits_sharpe = sharpe_ratio(profits, risk_free=0, period='daily', annualization=365)#525600)
    sortino_ratio_ = sortino_ratio(profits, required_return=0, period='daily', annualization=365)#525600)
    print(  '|org:',round(stdearn_sharpe,6),'|stg:',round(profits_sharpe,6),'|sortino_ratio:',sortino_ratio_)
    return

    profits = data['ocdiff%']*(data[prefix+'-period'].astype(int)*2-1)
    profits_sharpe = round(np.mean(profits)/np.std(profits)*np.sqrt(525600),6)
    stdearn = data['ocdiff%']
    stdearn_sharpe = round(np.mean(stdearn)/np.std(stdearn)*np.sqrt(525600),6)
    # print(profits)
    # print(stdearn)
    # print(data[prefix+'-period'])
    # exit()
    print('\n|',stdearn_sharpe,profits_sharpe,'\t|',prefix,'profits:\t',round(np.mean(profits),6),round(np.std(profits),6))
    sp_ratio = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=525600)
    print(sp_ratio)
    sp_ratio = sharpe_ratio(profits, risk_free=0, period='daily', annualization=525600)
    print(sp_ratio)
    st_ratio = sortino_ratio(profits, required_return=0, period='daily', annualization=525600)
    print(st_ratio)

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
    data = groupdata(data,length=30)#1440)#3)
    data['ocdiff%'] = data['ocdiff']/data['open']*100
    #data = data[1000:3000]
    #return data
    delta = data['ocdiff%']
    # stctests([delta])
    # drawfunc([delta], st.acf,[-0.15,0.15])
    # drawfunc([delta],st.pacf,[-0.15,0.15])
    #plt.show()
    #exit()


    model = arch_model(delta, mean='Zero', vol='GARCH', p=1, q=1)
    model_fit = model.fit()
    print(model_fit.summary)
    # #yhat = model_fit.forecast(horizon=1)
    # preds = model_fit.forecast()#align='target')
    # preds = preds.volatility
    # print(preds)
    # #exit()
    drawdata([delta,model_fit.conditional_volatility,-model_fit.conditional_volatility],'preds',figsize=(16,16))
    #fig = model_fit.hedgehog_plot()
    #fig = model_fit.plot()



    plt.show()
    exit()
    if 1:
        size = int(len(delta)*0.996)
        train, tests = delta[0:size], delta[size:len(delta)]
        # order = st.arma_order_select_ic(train,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
        # print(order)
        # exit()
        model = ARMA(train, order=(6,5))
        model_fit = model.fit(disp=-1, method='css')
        preds = model_fit.predict()
        resid = pd.Series(model_fit.resid)
        print(delta)
        print(train)
        print(resid)
        print(tests)
        fores = model_fit.forecast(len(tests))
        print(fores)
        fores = pd.Series(fores[0],index=tests.index)
        print(fores)
        error = tests-fores
        print(error)
        stctests([delta,preds,resid,fores,error])
        #drawfunc([delta,preds,resid,fores,error], st.acf,[-0.15,0.15])
        #drawfunc([delta,preds,resid,fores,error],st.pacf,[-0.15,0.15])
        drawdata([delta,preds,resid,fores,error],'preds',figsize=(16,16))


        plt.show()
        exit()



        history = [x for x in train]
        predictions0 = list()
        predictions1 = list()
        predictions2 = list()
        # walk-forward validation
        for t in tqdm.tqdm(range(len(test))):
            model = ARMA(history, order=(0,1))
            #model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=-1)
            output = model_fit.forecast(10)
            predictions0.append(output[0][0])
            predictions1.append(output[0][4])
            predictions2.append(output[0][9])
            history.append(test[t])
            #print('predicted=%f, expected=%f' % (output[0], test[t]))
        # evaluate forecasts
        rmse0 = np.sqrt(mean_squared_error(test, predictions0))
        print('Test RMSE0: %.3f' % rmse0)
        rmse1 = np.sqrt(mean_squared_error(test, predictions1))
        print('Test RMSE1: %.3f' % rmse1)
        rmse2 = np.sqrt(mean_squared_error(test, predictions2))
        print('Test RMSE2: %.3f' % rmse2)
        # plot forecasts against actual outcomes
        plt.plot(test)
        plt.plot(predictions0, color='red')
        plt.plot(predictions1, color='orange')
        plt.plot(predictions2, color='green')
        plt.show()
        exit()





    infolength = 200
    preds = [0]*infolength
    resid = [0]*infolength
    for t in tqdm.tqdm(range(len(delta)-infolength)):
        model = ARIMA(delta[t:t+infolength], order=(3,0,3))
        model_fit = model.fit()
        resid.append(model_fit.resid.mean())
        output = model_fit.forecast(1)
        preds.append(output.values[0])
    preds = pd.Series(preds,index=delta.index)
    resid = pd.Series(resid,index=delta.index)
    error = delta-preds
    stctests([delta,preds,error,resid])
    drawfunc([delta,preds,error,resid], st.acf,[-0.15,0.15])
    drawfunc([delta,preds,error,resid],st.pacf,[-0.15,0.15])
    drawdata([delta,preds,error,resid],'preds',figsize=(16,16))
    data['preds'] = preds
    return data
    # rs = []
    # for inum in range(100):
    #     series = [random.gauss(0.0, 1.0) for i in range(300*(inum+1))]
    #     series = pd.Series(series)
    #     r = acorr_ljungbox(series, lags=[1])
    #     print(r)
    # exit()

    # series = [random.gauss(0.0, 1.0) for i in range(3000)]
    # series = pd.Series(series)

    # data['ocmidlog'] = np.log(data['ocmid'])*1000
    # data['deltap'] = data['ocmidlog']-data['ocmidlog'].shift(periods=1,fill_value=data['ocmidlog'][0])
    # #data['deltap2']= data['deltap'].shift(periods=10,fill_value=0)
def main():
    measurements = 'bfs.autogen.klinehistory'
    instrument_id= 'BINANCE_SPOT_ETH-USDT'
    tags = 'open,close,high,low,count'#amount,amount_long,count,count_long,volumn,volumn_base,volumn_long
    headdatetime = str(int(datetime.datetime.strptime('20.12.2021 00:00:00,00','%d.%m.%Y %H:%M:%S,%f').timestamp()*1000))+'ms'
    taildatetime = str(int(datetime.datetime.strptime('30.12.2021 00:00:00,00','%d.%m.%Y %H:%M:%S,%f').timestamp()*1000))+'ms'
    querycommand = 'select '+tags+' from '+measurements+' where '+'time<'+taildatetime+' and '+'time>'+headdatetime+' and '+"instrument_id='"+instrument_id+"'"
    data = getdata(host='47.57.72.27', port=8086, querycommand=querycommand)
    # instrument_id= 'BINANCE_SPOT_ETH-USDT'
    # querycommand = 'select '+tags+' from '+measurements+' where '+'time<'+taildatetime+' and '+'time>'+headdatetime+' and '+"instrument_id='"+instrument_id+"'"
    # data2 = getdata(host='47.57.72.27', port=8086, querycommand=querycommand)

    # plt.plot(data['open'],color='red',alpha=1.,linewidth=.5,linestyle='--')
    # plt.plot(data2['open'],color='red',alpha=1.,linewidth=.5,linestyle='--')
    # plt.show()
    # exit()

    data = statics(data)
    indicator   = 'EMA'#'SMA'
    lengths1    = [7]#[480]#[1800]#[500]
    lengths2    = [3]#[30]#[11]#,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]#[200]
    parameters  = [30]#[1440]#[3600]#[1200]
    period      = 50
    indicators(data,indexies=sorted(list(dict.fromkeys(lengths1+lengths2+parameters))),indicator=indicator)
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
# import matplotlib.animation as animation
# frames = []
# fig = plt.figure()
# plt.axis('off')
# for i in tqdm.tqdm(range(len(data))):
#     img = drawheatmap(data[i],malongs,mashorts)#,True,str(i))
#     frames.append([plt.imshow(img,animated=True)])
# ani = animation.ArtistAnimation(fig, frames)#, interval=50, blit=True, repeat_delay=1000)
# ani.save('movie.mp4')
#     #data = yf.download('SPY', period='7d', interval='1m')#period='7d' 'SPY AAPL' EURUSD=X ETH-USD
#     #data = yf.download('SPY', period='15y', interval='1d')#period='7d' 'SPY AAPL' EURUSD=X ETH-USD
#     data = yf.download('SPY', start='1994-01-01', end='2002-12-30', interval='1d')#period='7d' 'SPY AAPL' EURUSD=X ETH-USD
#     mpf.plot(data,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)
#     # exit()
#     # data.index = pd.DatetimeIndex(data.index)
#     # idx = pd.date_range('2021-12-27 15:00:00+00:00', '2021-12-28 15:00:00+00:00', freq='1min')
#     # data = data.reindex(idx, method='ffill')#fill_value=0)
#     data = data.reset_index()
#     data = data.rename(columns={'index':'time','Open':'open','Close':'close','High':'high','Low':'low','Volume':'count','Adj Close':'close-adj'})
#     data['count'] = data['count']/100000

#     #data['ocdiff'] = data['close']-data['open']
#     #data['ocdiff%']= data['ocdiff']/data['open']*100
#     data['ocdiff'] = data['close'].diff()
#     data['close'] = data['close'].shift()
#     data['ocdiff%']= data['ocdiff']/data['close']*100
#     data = data[1:]
#     # debug(data,30)
#     # exit()

# #    profits = data['ocdiff%']*(data[prefix+'-period'].astype(int)*2-1)
# #    profits_sharpe = round(np.mean(profits)/np.std(profits)*np.sqrt(525600/len(data)/3),6)
#     stdearn = data['ocdiff%']
#     print(len(data))
#     #stdearn_sharpe = round(np.mean(stdearn)/np.std(stdearn)*np.sqrt(525600),6)
#     stdearn_sharpe = round(np.mean(stdearn)/np.std(stdearn)*np.sqrt(252),6)
#     # print(profits)
#     # print(stdearn)
#     # print(data[prefix+'-period'])
#     # exit()
# #    print('\n|',stdearn_sharpe,profits_sharpe,'\t|',prefix,'profits:\t',round(np.mean(profits),6),round(np.std(profits),6))
#     print(stdearn_sharpe)
#     #sp_ratio = sharpe_ratio(stdearn, risk_free=0, period='daily', annualization=525600)
#     sp_ratio = sharpe_ratio(stdearn, risk_free=0, period='daily')#, annualization=525600/len(data)/3)
#     print(sp_ratio)
#     exit()

#     sp_ratio = sharpe_ratio(profits, risk_free=0, period='daily', annualization=525600/len(data)/3)
#     print(sp_ratio)
#     an_return = annual_return(profits,annualization=len(data))
#     print(an_return)
#     st_ratio = sortino_ratio(profits, required_return=0, period='daily', annualization=525600/len(data)/3)
#     print(st_ratio)


# import pandas as pd
# import numpy as np
# from functools import reduce
# buy  = [0,0,0,1,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,1,0,0,0,0, 0,0, 0,0,0,0,1,0,0,0]
# sell = [0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,0,0,0,0,0,0]
# data  = pd.DataFrame({'buy':buy,'sell':sell})
# data['signal'] = data['buy']+data['sell']
# a = data['signal']#[0,0,0,1,0,0,0,1,0,0,0,0,-1,0,0,1,0,-1,0,0,0,0,0,0,-1,0,0,0,1,0,0,0,0]
# b = [0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1]#[0,0,0,1,1,1,1,1,1,1,1,1, 0,0,0,1,1, 0,0,0,0,0,0,0, 0,0,0,0,1,1,1,1,1]

# def update_state(state,signal):
#     assert state in (0,1), "state={}".format(state)
#     assert signal in (0,1,-1), "signal={}".format(signal)
#     if signal == 1 or state == 1 and signal!=-1: return 1
#     else: return 0
# update_state_u = np.frompyfunc(update_state,2,1)
# update_state_acc = lambda x: update_state_u.accumulate(x,dtype=object).astype(int)

# print([(state,signal,update_state(state,signal)) for state in (0,1) for signal in (0,1,-1)])
# b2 = update_state_acc(a)
# print(b2)
# print(b2 == b)
# exit()

# data['signal'] = data['buy']+data['sell']
# data['buycum'] = data['buy'].cumsum()
# data['sellcum']= data['sell'].cumsum()
# data['signcum']= data['signal'].cumsum()
# print(data)
# exit()

            # init_capital = 100000
            # signals = pd.DataFrame(index=data.index)
            # signals['flag']   = data[prefix+'-positive'].astype(int)
            # signals['signal'] = signals['flag'].diff().fillna(signals['flag'][0])

            # print(signals.head(100))

            # # def gen_positions(signals):
            # #     positions = signals['flag']*1000
            # #     return positions

            # # def trade_positions(signals):
            # #     positions = signals['signal']*1000
            # #     return positions

            # capital = pd.DataFrame(index=signals.index)
            # #capital = pd.DataFrame(index=data[data[prefix+'-optime']==True].index)
            # capital['hold'] = signals['flag']*data['close']
            # capital['middle'] = signals['signal']*data['close']
            # capital['middle2'] = capital['middle'].cumsum()
            # capital['rest'] = init_capital-capital['middle2']
            # capital['total'] = capital['hold']+capital['rest']
            # capital['return'] = capital['total'].pct_change().fillna(capital['total'][0]/init_capital-1)
            # print(capital.head(100))

# def getsignal2(data,malongs,mashorts):
#     starttime = time()
#     print("signal start")
#     for length in sorted(list(dict.fromkeys(malongs+mashorts))):
#         data['sma'+str(length)]=talib.SMA((data['open']+data['close'])/2,length)
#     for malong in tqdm.tqdm(malongs):
#         for mashort in tqdm.tqdm(mashorts):
#             if mashort>=malong: continue
#             prefix = str(mashort)+'-'+str(malong)
#             data[prefix]=data['sma'+str(mashort)]-data['sma'+str(malong)]
#             data[prefix+'-diff']        = data[prefix].diff()
#             data[prefix+'-positive']    =(data[prefix]>0)
#             data[prefix+'-signal']      = data[prefix+'-positive'].diff()
#             data.loc[0,prefix+'-signal']  = False
#             data[prefix+'-signal']        = data[prefix+'-signal'].astype(bool)
#             data[prefix+'-buysignal']   = data[prefix+'-positive']&data[prefix+'-signal']
#             data[prefix+'-buytime']     = data[prefix+'-buysignal'].shift(periods=1,fill_value=False)

#             data[prefix+'-selltime']    = data[prefix+'-buytime'].shift(periods=mashort//2,fill_value=False)
#             data[prefix+'-buyprice']    = data[prefix+'-buytime'] *data['high']#low,high
#             data[prefix+'-sellprice']   = data[prefix+'-selltime']*data['low']
#             data[prefix+'-profit']      = data[prefix+'-sellprice']/(data[prefix+'-buyprice'].shift(periods=mashort//2,fill_value=0))-1
#             data[prefix+'-profit']        = data[prefix+'-profit'].replace(np.nan, 0)
#     print("signal got, used",time()-starttime,'s')
#     return data

# def drawcapitals(data,malongs,mashorts,savefig=False,figsize=(16,9)):
#     starttime = time()
#     print("drawcapitals start")
#     folder = './capitals/'
#     os.makedirs(folder,exist_ok=True)
#     for malong in tqdm.tqdm(malongs):
#         figname = 'smalong'+str(malong)+'shorts'+'-'.join([str(mashort) for mashort in mashorts])
#         fig = plt.figure(figname, figsize=figsize)
#         colors = cycle(COLORS)
#         for mashort in tqdm.tqdm(mashorts):
#             if mashort>=malong: continue
#             prefix = str(mashort)+'-'+str(malong)
#             data[prefix+'-hold'] = data[prefix+'-holdtime2']*data['ocmid']
#             data[prefix+'-flow'] = data[prefix+'-buyprice']+data[prefix+'-sellprice']
#             data[prefix+'-rest'] =-data[prefix+'-flow'].cumsum()
#             data[prefix+'-total']= data[prefix+'-hold']+data[prefix+'-rest']
#             color=next(colors)
#             plt.plot(data[prefix+'-total'],color=color,alpha=1.,linewidth=.71)
#         plt.plot(data['ocmid']-data['ocmid'][0],color='black',alpha=1.,linewidth=.71)
#         plt.gca().set_xticks([i for i in range(int(plt.xlim()[0]),int(plt.xlim()[1]),5000)])
#         plt.gca().set_yticks([i for i in range(int(plt.ylim()[0]),int(plt.ylim()[1]),1000)])
#         plt.grid(color='black',linewidth=.1)
#         if savefig: plt.savefig(folder+figname+'.png', dpi=200, facecolor="azure", bbox_inches='tight', pad_inches=0)
#         plt.close()
#     print("drawcapitals done, used",time()-starttime,'s')

# def drawshorts(data,malongs,mashorts,savefig=False,figsize=(16,9)):
#     starttime = time()
#     print("drawshorts start")
#     folder = './shorts/'
#     os.makedirs(folder,exist_ok=True)
#     for mashort in tqdm.tqdm(mashorts):
#         figname = 'sma'+str(mashort)+'x'+'-'.join([str(malong) for malong in malongs])
#         fig = plt.figure(figname, figsize=figsize)
#         plt.fill_between(data.index, data['low'], data['high'], facecolor='grey', alpha=0.75)
#         plt.fill_between(data.index, data['open'],data['close'],facecolor='black',alpha=0.75)
#         plt.plot(data['sma'+str(mashort)],color='black',alpha=1.,linewidth=.71)
#         colors = cycle(COLORS)
#         for malong in malongs:
#             if mashort>=malong: continue
#             color=next(colors)
#             plt.plot(data['sma'+str(malong)],color=color,alpha=1.,linewidth=.71)
#             prefix = str(mashort)+'-'+str(malong)
#             plt.scatter(data[data[prefix+'-buytime']==1].index, data[data[prefix+'-buytime']==1]['high'], color=color, marker='+', s=35)
#             plt.scatter(data[data[prefix+'-selltime']==1].index,data[data[prefix+'-selltime']==1]['low'], color=color, marker='.', s=35)
#         if savefig: plt.savefig(folder+figname+'.png', dpi=200, facecolor="azure", bbox_inches='tight', pad_inches=0)
#         plt.close()
#     print("drawshorts done, used",time()-starttime,'s')

# def prefixsgen(prefix,lists):
#     prefixs = []
#     for element1 in lists:
#         for element2 in lists:
#             if element1>=element2: continue
#             prefixs.append(prefix+'-'+str(element1)+'-'+str(element2))
#     return prefixs

# def drawtraces(data,indexi,indexj):
#     prefix = str(indexj)+'-'+str(indexi)
#     if prefix not in data.columns: return None,None
#     else:
#         for index, row in data.iterrows():
#             if row[prefix+'-buytime']==True:
#                 zeroprice = row[prefix+'-buyprice-long']
#                 dataslice = data[index-indexj:index+indexi][['low','high','open','close']]
#                 dataslice = (dataslice-zeroprice).reset_index()
#                 plt.fill_between(dataslice.index, dataslice['low'], dataslice['high'], facecolor='grey', alpha=0.75)
#                 plt.fill_between(dataslice.index, dataslice['open'],dataslice['close'],facecolor='black',alpha=0.75)
#         return None,prefix

# @timer
# def drawheatmaps(data,malongs,mashorts,period,savefig=False,figsize=(16,16),dpi=200,fps=2,maxrange=None):
#     videoname = 'heatmap'+'-'.join([str(mashort) for mashort in mashorts])+'x'+'-'.join([str(malong) for malong in malongs])+'period'+str(period)
#     vWriter = cv2.VideoWriter(videoname+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (figsize[0]*dpi,figsize[1]*dpi))
#     figname = 'heatmap'
#     data = [data[i:i+period] for i in range(0,len(data),period)]
#     for i in tqdm.tqdm(range(len(data))):
#         profitmatrix = getprofits(data[i],indexies=malongs,indexjes=mashorts)
#         profitmatrix = pd.DataFrame(profitmatrix,index=malongs,columns=mashorts)
#         fig = plt.figure(num=figname+str(i),figsize=figsize)
#         if maxrange==None: maxrange = profitmatrix.abs().max().max()
#         ax = sns.heatmap(profitmatrix, annot=True, fmt='.1f', cmap='coolwarm', vmin=-maxrange, vmax=maxrange)#jet,coolwarm_r #bar(left, height, width=0.8, bottom=None)
#         fig.canvas.draw()# redraw the canvas    #fig.axes[0].set_axis_on()
#         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)#, sep='')# convert canvas to image
#         img = img.reshape([fig.canvas.get_width_height()[0]*2,fig.canvas.get_width_height()[1]*2,3]) # 4 times larger...
#         if savefig: imageio.imwrite(figname+str(i)+'.jpg', img)
#         plt.close()
#         img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)# img is rgb, convert to opencv's default bgr
#         vWriter.write(img)
#     vWriter.release()












# np.random.seed(19680801)

# dt = 0.01
# t = np.arange(0, 30, dt)
# nse1 = np.random.randn(len(t))                 # white noise 1
# nse2 = np.random.randn(len(t))                 # white noise 2

# # Two signals with a coherent part at 10Hz and a random part
# s1 = np.sin(2 * np.pi * 10 * t) + nse1
# s2 = np.sin(2 * np.pi * 10 * t) + nse2



# fig = plt.figure(figsize=figsize)

# fig, axs = plt.subplots(2, 1)





# exit()
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t, s1, t, s2)
# axs[0].set_xlim(0, 2)
# axs[0].set_xlabel('time')
# axs[0].set_ylabel('s1 and s2')
# axs[0].grid(True)

# cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
# axs[1].set_ylabel('coherence')

# fig.tight_layout()
# plt.show()


# exit()

# @looper
# def indicators(data,indexi,indicator):
#     if indicator=='EMA': data[indicator+str(indexi)]=talib.EMA(data['ocmid'],indexi)
#     if indicator=='SMA': data[indicator+str(indexi)]=talib.SMA(data['ocmid'],indexi)
# @timer
# @looper2
# def getsignals(data,indexi,indexj,indicator):
#     if indexj>=indexi:# or indexj<indexi//105:
#         return
#     prefix = str(indexi)+'-'+str(indexj)
#     data[prefix]=data[indicator+str(indexj)]-data[indicator+str(indexi)]
#     data[prefix+'-diff']        = data[prefix].diff()
#     data[prefix+'-positive']    =(data[prefix]>0)
#     data[prefix+'-possignal']   = data[prefix+'-positive'].diff()
#     data.loc[0,prefix+'-possignal'] = False
#     data.loc[indexi-1,prefix+'-possignal'] = False
#     data[prefix+'-possignal']       = data[prefix+'-possignal'].astype(bool)
#     data[prefix+'-buysignal']   = data[prefix+'-positive']&data[prefix+'-possignal']#|
#     data[prefix+'-buytime']     = data[prefix+'-buysignal'].shift(periods=1,fill_value=False).astype(int)
#     data[prefix+'-negative']    =(data[prefix]<0)
#     data[prefix+'-negsignal']   = data[prefix+'-negative'].diff()
#     data.loc[0,prefix+'-negsignal'] = False
#     data.loc[indexi-1,prefix+'-negsignal'] = False
#     data[prefix+'-negsignal']       = data[prefix+'-negsignal'].astype(bool)
#     data[prefix+'-sellsignal']  = data[prefix+'-negative']&data[prefix+'-negsignal']
#     data[prefix+'-selltime']    = data[prefix+'-sellsignal'].shift(periods=1,fill_value=False).astype(int)
#     data[prefix+'-tradetime']   = data[prefix+'-buytime']-data[prefix+'-selltime']#.shift(periods=1,fill_value=0)
#     #data[prefix+'-holdtime']    = data[prefix+'-tradetime'].cumsum()

#     suffix = '-long'
#     def update_state_long(state,signal): # long trade
#         assert state in (0,1), "state={}".format(state)
#         assert signal in (0,1,-1), "signal={}".format(signal)
#         if signal == 1 or state == 1 and signal!=-1: return 1
#         else: return 0
#     update_state_long_u = np.frompyfunc(update_state_long,2,1)
#     update_state_long_acc = lambda x: update_state_long_u.accumulate(x,dtype=object).astype(int)
#     data[prefix+'-holdtime2'+suffix]   = update_state_long_acc(data[prefix+'-tradetime'])
#     data[prefix+'-tradetime2'+suffix]  = data[prefix+'-holdtime2'+suffix].diff()
#     data.loc[0,prefix+'-tradetime2'+suffix] = 0
#     data[prefix+'-buytime2'+suffix]    = data[prefix+'-tradetime2'+suffix].clip(lower=0)
#     data[prefix+'-selltime2'+suffix]   = data[prefix+'-tradetime2'+suffix].clip(upper=0)
#     data[prefix+'-buyprice'+suffix]    = data[prefix+'-buytime2'+suffix] *data['open']#low,high
#     data[prefix+'-sellprice'+suffix]   = data[prefix+'-selltime2'+suffix]*data['open']
#     # capital lines
#     data[prefix+'-hold'+suffix] = data[prefix+'-holdtime2'+suffix]*data['ocmid']
#     data[prefix+'-flow'+suffix] =-data[prefix+'-buyprice'+suffix]-data[prefix+'-sellprice'+suffix]
#     data[prefix+'-rest'+suffix] = data[prefix+'-flow'+suffix].cumsum()
#     data[prefix+'-total'+suffix]= data[prefix+'-hold'+suffix]+data[prefix+'-rest'+suffix]

#     suffix = '-short'
#     def update_state_short(state,signal): # short trade
#         assert state in (0,1), "state={}".format(state)
#         assert signal in (0,1,-1), "signal={}".format(signal)
#         if signal == -1 or state == 1 and signal!=1: return 1
#         else: return 0
#     update_state_short_u = np.frompyfunc(update_state_short,2,1)
#     update_state_short_acc = lambda x: update_state_short_u.accumulate(x,dtype=object).astype(int)
#     data[prefix+'-holdtime2'+suffix]   = update_state_short_acc(data[prefix+'-tradetime'])
#     data[prefix+'-holdtime3'+suffix]   = update_state_long_acc(-data[prefix+'-tradetime'])
#     data[prefix+'-tradetime2'+suffix]  = data[prefix+'-holdtime2'+suffix].diff()
#     data.loc[0,prefix+'-tradetime2'+suffix] = 0
#     data[prefix+'-buytime2'+suffix]    = data[prefix+'-tradetime2'+suffix].clip(lower=0)
#     data[prefix+'-selltime2'+suffix]   = data[prefix+'-tradetime2'+suffix].clip(upper=0)
#     data[prefix+'-buyprice'+suffix]    = data[prefix+'-buytime2'+suffix] *data['open']#low,high
#     data[prefix+'-sellprice'+suffix]   = data[prefix+'-selltime2'+suffix]*data['open']
#     # capital lines
#     data[prefix+'-hold'+suffix] =-data[prefix+'-holdtime2'+suffix]*data['ocmid']
#     data[prefix+'-flow'+suffix] = data[prefix+'-buyprice'+suffix]+data[prefix+'-sellprice'+suffix]
#     data[prefix+'-rest'+suffix] = data[prefix+'-flow'+suffix].cumsum()
#     data[prefix+'-total'+suffix]= data[prefix+'-hold'+suffix]+data[prefix+'-rest'+suffix]

#     data[prefix+'-total'] = data[prefix+'-total-long']+data[prefix+'-total-short']




    # # series = [random.gauss(0.0, 1.0) for i in range(130000)]
    # # series = pd.Series(series)
    # # result = acf(series)
    # # print(result)
    # # #plt.plot(result)
    # # autocorrelation_plot(series)
    # # plt.show()
    # # exit()


    # data['deltap'] = data['ocmid']-data['ocmid'].shift(periods=30,fill_value=data['ocmid'][0])
    # data['deltap2']= data['deltap'].shift(periods=60,fill_value=0)
    # print(data.head(100))
    # result = acf(data['deltap'])
    # print(result)
    # #plt.plot(result)
    # autocorrelation_plot(data['deltap'])
    # plt.show()
    # exit()


    # result = kpss(data['deltap'])
    # print(result)
    # plt.scatter(data['deltap2'],data['deltap'], color='black', alpha=.021, marker='.', s=35)
    # plt.grid()
    # plt.show()
    # exit()

# def statics(data):
#     data = groupdata(data,length=5)
#     data = data[:2000]
#     data['ocdiff%'] = data['ocdiff']/data['open']*100
#     delta = data['ocdiff%']

#     stctests([delta])
#     # drawfunc([delta], st.acf,[-0.15,0.15])
#     # drawfunc([delta],st.pacf,[-0.15,0.15])
#     #fig = plt.figure('auto',figsize=(8,8))
#     #ax = autocorrelation_plot(delta)
#     #plt.show()
#     #exit()

#     #order = st.arma_order_select_ic(delta,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
#     #print(order)

#     infolength = 500
#     preds = [0]*infolength
#     for t in tqdm.tqdm(range(len(delta)-infolength)):
#         model = ARMA(delta[t:t+infolength], order=(1,0))
#         model_fit = model.fit(disp=-1)
#         output = model_fit.forecast(1)
#         preds.append(output[0][0])
#     data['preds'] = preds
#     resid = delta - preds
#     stctests([delta,preds,resid])
#     # drawfunc([delta,preds,resid], st.acf,[-0.15,0.15])
#     # drawfunc([delta,preds,resid],st.pacf,[-0.15,0.15])
#     # drawdata([delta,preds,resid],'preds',figsize=(16,16))

#     return
#     # rs = []
#     # for inum in range(100):
#     #     series = [random.gauss(0.0, 1.0) for i in range(300*(inum+1))]
#     #     series = pd.Series(series)
#     #     r = acorr_ljungbox(series, lags=[1])
#     #     print(r)
#     # exit()

#     # series = [random.gauss(0.0, 1.0) for i in range(3000)]
#     # series = pd.Series(series)

#     data['ocmidlog'] = np.log(data['ocmid'])*1000
#     data['deltap'] = data['ocmidlog']-data['ocmidlog'].shift(periods=1,fill_value=data['ocmidlog'][0])
#     #data['deltap2']= data['deltap'].shift(periods=10,fill_value=0)
#     print(data.head(10))
#     delta = data['deltap']
#     delta = delta[:5000]

#     model = ARMA(delta, order=(8,15))
#     result_arma = model.fit(disp=-1, method='css')
#     resid = pd.DataFrame(result_arma.resid)
#     print(resid)

#     k = kpss(delta)
#     r = acorr_ljungbox(delta, lags=list(range(1,10,2)))#, boxpierce=True)
#     print(k,r)
#     k = kpss(resid)
#     r = acorr_ljungbox(resid, lags=list(range(1,10,2)))#, boxpierce=True)
#     print(k,r)
#     # k = kpss(series)
#     # r = acorr_ljungbox(series, lags=list(range(1,10,2)))#, boxpierce=True)
#     # print(k,r)

#     pred = result_arma.predict()
#     fig = plt.figure('pred',figsize=(16,16))
#     plt.plot(delta,linewidth=.21)
#     plt.plot(pred,color='red',linewidth=.21)
#     plt.plot(resid,color='green',linewidth=.21)
#     #acc_pred = pred.cumsum()
#     #plt.plot(data['ocmidlog'][:10000]-data['ocmidlog'][0], color='blue')
#     #plt.plot(acc_pred, color='red')
#     #plt.scatter(delta,pred, color='black', alpha=.021, marker='.', s=35)
#     plt.grid()
#     #plt.show()
#     #exit()

#     fig = plt.figure('acf',figsize=(8,8))
#     deltapacf = acf(delta)
#     print(deltapacf)
#     plt.plot(deltapacf)
#     residpacf = acf(resid)
#     print(residpacf)
#     plt.plot(residpacf,color='green')
#     plt.ylim([-0.15,0.15])
#     plt.grid()

#     fig = plt.figure('pacf',figsize=(8,8))
#     deltapacf = pacf(delta)
#     print(deltapacf)
#     plt.plot(deltapacf)
#     residpacf = pacf(resid)
#     print(residpacf)
#     plt.plot(residpacf,color='green')
#     plt.ylim([-0.15,0.15])
#     plt.grid()
#     plt.show()
#     exit()

#     predict_ts = result_arma.predict()
#     print(predict_ts)
#     #plt.plot(delta)
#     #plt.plot(predict_ts)
#     acc_predict = predict_ts.cumsum()
#     plt.plot(data['ocmidlog'][:10000]-data['ocmidlog'][0], color='blue')
#     plt.plot(acc_predict, color='red')
#     #plt.scatter(delta,predict_ts, color='black', alpha=.021, marker='.', s=35)
#     plt.grid()
#     plt.show()
#     exit()

#     X = data['deltap'][:500].values*1000
#     size = int(len(X) * 0.66)
#     train, test = X[0:size], X[size:len(X)]
#     # order = st.arma_order_select_ic(train,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
#     # print(order)
#     # exit()
#     history = [x for x in train]
#     predictions0 = list()
#     predictions1 = list()
#     predictions2 = list()
#     # walk-forward validation
#     for t in tqdm.tqdm(range(len(test))):
#         model = ARMA(history, order=(0,1))
#         #model = ARIMA(history, order=(5,1,0))
#         model_fit = model.fit(disp=-1)
#         output = model_fit.forecast(10)
#         predictions0.append(output[0][0])
#         predictions1.append(output[0][4])
#         predictions2.append(output[0][9])
#         history.append(test[t])
#         #print('predicted=%f, expected=%f' % (output[0], test[t]))
#     # evaluate forecasts
#     rmse0 = np.sqrt(mean_squared_error(test, predictions0))
#     print('Test RMSE0: %.3f' % rmse0)
#     rmse1 = np.sqrt(mean_squared_error(test, predictions1))
#     print('Test RMSE1: %.3f' % rmse1)
#     rmse2 = np.sqrt(mean_squared_error(test, predictions2))
#     print('Test RMSE2: %.3f' % rmse2)
#     # plot forecasts against actual outcomes
#     plt.plot(test)
#     plt.plot(predictions0, color='red')
#     plt.plot(predictions1, color='orange')
#     plt.plot(predictions2, color='green')
#     plt.show()
#     exit()

#     order = st.arma_order_select_ic(delta,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
#     print(order)
#     exit()

#     r = kpss(delta)
#     print(r)
#     exit()

#     result = pacf(data['deltap'])
#     print(result)
#     plt.plot(result)
#     plt.ylim([-0.15,0.15])
#     plt.grid()
#     plt.show()
#     exit()

#     data = [data[i:i+20000] for i in range(0,100000,20000)]
#     for datai in data:
#         result = acf(datai['deltap'])
#         print(result)
#         #exit()
#         plt.plot(result)
#     plt.ylim([-0.05,0.05])
#     #ax = autocorrelation_plot(data['deltap'])
#     plt.show()
#     exit()

#     plt.scatter(data['deltap2'],data['deltap'], color='black', alpha=.021, marker='.', s=35)
#     plt.grid()
#     plt.show()
#     exit()

    # model = ARMA(delta, order=(8,15))
    # result_arma = model.fit(disp=-1, method='css')
    # resid = pd.DataFrame(result_arma.resid)
    # print(resid)
    # pred = result_arma.predict()

    # X = data['deltap'][:500].values*1000
    # size = int(len(X) * 0.66)
    # train, test = X[0:size], X[size:len(X)]
    # history = [x for x in train]
    # predictions0 = list()
    # predictions1 = list()
    # predictions2 = list()
    # for t in tqdm.tqdm(range(len(test))):
    #     model = ARMA(history, order=(0,1))
    #     #model = ARIMA(history, order=(5,1,0))
    #     model_fit = model.fit(disp=-1)
    #     output = model_fit.forecast(10)
    #     predictions0.append(output[0][0])
    #     predictions1.append(output[0][4])
    #     predictions2.append(output[0][9])
    #     history.append(test[t])
    # rmse0 = np.sqrt(mean_squared_error(test, predictions0))
    # print('Test RMSE0: %.3f' % rmse0)
    # rmse1 = np.sqrt(mean_squared_error(test, predictions1))
    # print('Test RMSE1: %.3f' % rmse1)
    # rmse2 = np.sqrt(mean_squared_error(test, predictions2))
    # print('Test RMSE2: %.3f' % rmse2)

    # s = pd.Series({'2021-12-27 15:00:00+00:00': 2,
    #                '2021-12-27 15:01:00+00:00': 10,
    #                '2021-12-27 15:06:00+00:00': 5,
    #                '2021-12-27 15:08:00+00:00': 1})
    # s.index = pd.DatetimeIndex(s.index)
    # idx = pd.date_range('2021-12-27 15:00:00+00:00', '2021-12-27 15:10:00+00:00', freq='1min')
    # s = s.reindex(idx, method='ffill')
    # print(s)
    # exit()
    # data = yf.download('ETH-USD', start='2021-12-28', end='2021-12-29', interval='1m')
    # data = data[40:60]
    # data.info()
    # data.index = pd.DatetimeIndex(data.index)#, freq='1m')
    # #print(data.index)
    # print(data.index.freq)
    # print(data)
    # idx = pd.date_range('2021-12-27 15:40:00+00:00', '2021-12-27 15:59:00+00:00', freq='1min')
    # data = data.reindex(idx, method='ffill')#fill_value=0)
    # print(data)
    # exit()

    # measurements = 'bfs.autogen.klinehistory'
    # instrument_id= 'BINANCE_SPOT_ETH-USDT'
    # tags = 'open,close,high,low,count'#amount,amount_long,count,count_long,volumn,volumn_base,volumn_long
    # from datetime import datetime
    # #dt_obj = datetime.strptime('01.12.2020 00:00:00,00','%d.%m.%Y %H:%M:%S,%f')
    # dt_obj = datetime.strptime('28.12.2021 00:00:00,00','%d.%m.%Y %H:%M:%S,%f')
    # headdatetime = str(int(dt_obj.timestamp()*1000))+'ms'
    # dt_obj = datetime.strptime('29.12.2021 00:00:00,00','%d.%m.%Y %H:%M:%S,%f')
    # taildatetime = str(int(dt_obj.timestamp()*1000))+'ms'
    # querycommand = 'select '+tags+' from '+measurements+' where '+'time<'+taildatetime+' and '+'time>'+headdatetime+' and '+"instrument_id='"+instrument_id+"'"
    # data = getdata(host='47.57.72.27', port=8086, querycommand=querycommand)
    # # data = groupdata(data,length=60)
    # #debug(data,1440)
    # debug(data,60)

    # fig = plt.figure('compare',figsize=(16,16))
    # plt.plot(data['open'],color='red',alpha=1.,linewidth=.5)
    # plt.plot(data['close'],color='green',alpha=1.,linewidth=.5)
    # plt.plot(data['high'],color='black',alpha=1.,linewidth=.5)
    # plt.plot(data['low'],color='black',alpha=1.,linewidth=.5)

    # #data = yf.download('ETH-USD', start='2021-10-25', end='2021-11-01', interval='1h')#'SPY AAPL' EURUSD=X ETH-USD
    # #data = yf.download('ETH-USD',period='7d',interval='1m')
    # data = yf.download('ETH-USD', start='2021-12-28', end='2021-12-29', interval='1m')
    # #mpf.plot(data,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)
    # data.info()
    # debug(data,60)
    # data.index = pd.DatetimeIndex(data.index)
    # debug(data,60)
    # idx = pd.date_range('2021-12-27 15:00:00+00:00', '2021-12-28 15:00:00+00:00', freq='1min')
    # data = data.reindex(idx, method='ffill')#fill_value=0)
    # debug(data,60)

    # data = data.reset_index()
    # data = data.rename(columns={'index':'time','Open':'open','Close':'close','High':'high','Low':'low','Volume':'count','Adj Close':'close-adj'})
    # data['count'] = data['count']/100000
    # data.info()
    # #debug(data,1440)
    # debug(data,60)

    # plt.plot(data['open'],color='red',alpha=1.,linewidth=.5,linestyle='--')
    # plt.plot(data['close'],color='green',alpha=1.,linewidth=.5,linestyle='--')
    # plt.plot(data['high'],color='black',alpha=1.,linewidth=.5,linestyle='--')
    # plt.plot(data['low'],color='black',alpha=1.,linewidth=.5,linestyle='--')
    # plt.show()

    # exit()


