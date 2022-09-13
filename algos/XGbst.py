import numpy as np
from pprint import pprint
import torch,algos,os,pickle
import importlib
import xgboost as xgb
import seaborn as sns
#sns.pairplot(pd_data, x_vars=['x1','x2'], y_vars='y',kind="reg", size=5, aspect=0.7)
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class Algo(algos.PTAlgo):
    def __init__(self,obs_space,act_space,args):
        algos.PTAlgo.__init__(self,obs_space,act_space,args)
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = torch.device("cuda:0")
        params ={
            'learning_rate':0.1,
            'max_depth':5,#10,                # 构建树的深度，越大越容易过拟合
            #'num_boost_round':2000,
            'objective':'reg:squarederror',  # 线性回归问题 #'objective': 'reg:linear',      # 线性回归问题，早期版本的参与，将被reg:squarederror替换
            #'random_state':7,
            #'gamma':0,
            #'subsample':0.8,
            'colsample_bytree':0.3,#0.8,
            'reg_alpha':10,#0.005,
            'n_estimators':10,#1000,
            #'eval_metric':['logloss','rmse','mae'],   #分类有“auc”
            #'eta':0.3                      #为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
            'verbosity':3,
        }
        self.model = xgb.XGBRegressor(**params)#xgb.XGBRegressor()#xgb.XGBClassifier()#LinearRegression()
        if self.act_space.__class__.__name__ == 'Discrete': self.action_shape = 1
        else:                                               self.action_shape = self.act_space.shape[0]
        self.fitted= False
        self.x, self.y = [], []
    def get_action(self, inputs, explore): # inputs: batch stack width height channel
        if self.fitted:
            inputs = inputs.reshape(inputs.shape[0],-1)
            action = self.model.predict(inputs)
            action = action.squeeze()
            info_p = {}
        else:
            action = np.zeros(inputs.shape[0])
            info_p = {}
        return action, info_p
    def update(self, crt_step, max_step, info_in):
        testing = info_in['mb_info'][0]['testing']
        if not self.fitted:
            if testing:
                if len(self.x)!=0:
                    print('updating...')
                    # print(self.x.shape)
                    # print(self.y.shape)
                    # exit()
                    model_fit=self.model.fit(self.x, self.y)
                    print('self.model',self.model)
                    pprint(vars(self.model))
                    print('model_fit',model_fit)
                    pprint(vars(model_fit))
                    self.fitted = True
                    exit()
            else:
                x = info_in['mb_obs'][0]
                y = info_in['mb_info'][0]['labels']
                x = x.reshape(x.shape[0],-1)
                y = y[:,np.newaxis]
                if len(self.x)==0:
                    self.x = x
                    self.y = y
                else:
                    self.x = np.concatenate((self.x, x), axis=0)
                    self.y = np.concatenate((self.y, y), axis=0)
        else:
            print('fitted!',crt_step,max_step)

    def save(self,name,prefix=''):
        foldername = self.args.exp_dir+'models/'
        os.makedirs(foldername,exist_ok=True)
        pickle.dump(self.model, open(foldername+prefix+name, 'wb'))
    def load(self,prefix='',folder=''):
        try:
            print('load_model folder:',folder)
            print('load_model prefix:',prefix)
            if folder=='': flist = glob.glob(self.args.exp_dir+'models/'+prefix+'*')
            else:          flist = glob.glob(folder+prefix+'*')
            print('load_model flist:',[fname.split('/')[-1] for fname in flist])
            flist = [ffile for ffile in flist if os.path.isfile(ffile)]
            print('load_model files:',[fname.split('/')[-1] for fname in flist])
            ffile = max(flist, key=os.path.getmtime)#ctime)
            print('load_model ffile:',ffile.split('/')[-1])
            self.model = pickle.load(open(ffile, 'rb'))
            return ffile
        except:
            print('Error when trying to load model...Skipped.')
            print('load_model folder:',folder)
            print('load_model prefix:',prefix)
            return None

def fAlgo(obs_space,act_space,args):
    algo = Algo(obs_space,act_space,args)
    return algo


# param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)


