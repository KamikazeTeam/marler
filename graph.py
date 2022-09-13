import numpy as np
import argparse,json,easydict,random,os,time,pprint,sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import cycle
def fig_curves(expfolder,args_draw,args):
    realdata_alpha,realdata_width,realmeandata_alpha,realmeandata_width = 0.01,0.01,0.03,0.01
    if args.env_mode == 'supervise': args.env_num,args_draw.avgnum,realdata_alpha,realdata_width = 1,1,1.0,0.5
    COLORS = cycle(['red','orange','green','cyan','blue','purple'])#,'black'])
    filename = expfolder+'/'+args_draw.subfold+'/'+args_draw.namepfx
    print('filename',filename)
    lines = [[line for line in open(filename+str(i)+'_'+str(args_draw.teamdraw),"r").read().splitlines() if len(line)!=0] for i in range(args.env_num)]
    #print(len(lines))
    #print(len(lines[0]))
    #for i in range(args.env_num):
    #    print(i,len(lines[i]))
    plt.figure(figsize=(16,9))
    regularxmeans, regularymeans = [], []
    for j in range(len(lines[0])):
        #if j%args_draw.groupnums!=args_draw.groupdraw: continue
        if len(lines[0][j])==0: continue#skip empty line
        color=next(COLORS)
        xytuples = []
        for i in range(args.env_num):
            try:
                records = lines[i][j].split("|")[:-1]#[:int(args.max_episodes)]
            except:
                print(i,j)
                exit()
            x, y = [], []
            for record in records:
                xe, ye = int(record.split(',')[0]), float(record.split(',')[1])
                x.append(xe)
                y.append(ye)
                xytuples.append((xe,ye))
            xmean = [x[k] for k in range(len(x))]#-args_draw.avgnum+1)]
            ymean = [np.mean(y[max(0,l+1-args_draw.avgnum):l+1]) for l in range(len(y))]#[np.mean(y[l:l+args_draw.avgnum]) for l in range(len(y))]#-args_draw.avgnum+1)]
            plt.plot(x,y,color=color,alpha=realdata_alpha,linewidth=realdata_width)
            #print(args_draw.subfold,i)
            #print('xmean',xmean)
            #print('ymean',ymean)
            plt.plot(xmean,ymean,color=color,alpha=realmeandata_alpha,linewidth=realmeandata_width)
        if args_draw.namepfx=='test_': print(xytuples)
        sortedxytuples = sorted(xytuples)
        sortedxytuplesx= [xytuple[0] for xytuple in sortedxytuples]
        sortedxytuplesy= [xytuple[1] for xytuple in sortedxytuples]
        if args_draw.namepfx=='test_': print(np.array(sortedxytuplesy).mean())
        #sortedxytuplesxmean = [sortedxytuplesx[k] for k in range(len(sortedxytuplesx)-args_draw.avgnum+1)]
        #sortedxytuplesymean = [np.mean(sortedxytuplesy[l:l+args_draw.avgnum]) for l in range(len(sortedxytuplesy)-args_draw.avgnum+1)]
        #plt.plot(sortedxytuplesx,sortedxytuplesy,color=color,alpha=0.5,linewidth=0.1)######
        #plt.plot(sortedxytuplesxmean,sortedxytuplesymean,color=color,alpha=0.8,linewidth=0.3)
        regularx, regulary = [], []
        for istep in range(int(args.start_step),int(args.start_step+args.max_steps),int(args.max_steps/200)):######
            index = next((index for index,value in enumerate(sortedxytuplesx) if value>istep), len(sortedxytuplesx)-1)
            if index!=0:# continue
                regularx.append(istep)
                regulary.append(np.mean(sortedxytuplesy[max(0,index-100):index]))######
            else:
                regularx.append(istep)
                regulary.append(0.0)
        plt.plot(regularx,regulary,color=color,alpha=0.8,linewidth=0.3)######
        regularxmeans = regularx
        regularymeans.append(regulary)
    regularymeansmean= np.array(regularymeans).mean(axis=0)
    regularymeansvar = np.array(regularymeans).std(axis=0)
    plt.plot(regularxmeans,regularymeansmean,color='black',alpha=1.0,linewidth=0.3)
    plt.fill_between(regularxmeans, regularymeansmean-regularymeansvar, regularymeansmean+regularymeansvar,facecolor='black',alpha=0.5)
    max_steps = int(args.max_steps+args.start_step)#int(args.max_steps)*int(args.roll_num)*int(args.env_num)
    maxscore, minscore, n = float(args.minmax_score.split(',')[1]), float(args.minmax_score.split(',')[0]), 10
    axes = plt.gca()
    axes.set_xticks(np.arange(0,max_steps,max_steps/n))
    plt.xlim([0,max_steps])
    #print(maxscore,minscore)
    if maxscore>0 or minscore<0:
        #print('setting ylim')
        diffscore = (maxscore - minscore)/n
        axes.set_yticks(np.arange(minscore-diffscore,maxscore+diffscore,diffscore))
        plt.ylim([minscore-diffscore,maxscore+diffscore])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(labelsize=8)
    plt.grid(linewidth=0.1)
    figname = expfolder+args_draw.subfold+args_draw.namepfx#+str(args.numparas)
    # figname = figname.replace('full','f').replace('none','n').replace('prev','p').replace('imagine','img')
    # print('figname',figname) # print('figname[:252]',figname[:252])
    plt.savefig(figname+str(args_draw.teamdraw)+str(args_draw.teamnums)+'.png', dpi=200, facecolor="azure", bbox_inches='tight')#pad_inches=0)
    plt.close()
    fall = open(args_draw.folder+'_'+args.env_name,'a')
    print(figname,end='|',file=fall)
    for data in regularxmeans:
        print(data,end=',',file=fall)
    print('',end='|',file=fall)
    for data in regularymeansmean:
        print(data,end=',',file=fall)
    print('',end='|',file=fall)
    for data in regularymeansvar:
        print(data,end=',',file=fall)
    print('',file=fall)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw')
    parser.add_argument('--folder', default='results', help='')
    parser.add_argument('--subfold', default='rewards', help='')
    parser.add_argument('--namepfx', default='', help='')
    parser.add_argument('--minmax', default='', help='')
    parser.add_argument('--avgnum', type=int, default=10, help='')
    parser.add_argument('--teamnums', type=int, default=0, help='')
    parser.add_argument('--teamdraw', type=int, default=0, help='')
    args_draw = parser.parse_args()
    folder = args_draw.folder+'/'
    files = os.listdir(folder)
    files.sort()
    for file in files:
        if file[-4:]!='args': continue
        args = easydict.EasyDict()
        with open(folder+file, 'r') as f:
            args.__dict__ = json.load(f)
        if args_draw.minmax!="": args.minmax_score = args_draw.minmax
        fig_curves((folder+file)[:-5],args_draw,args)
