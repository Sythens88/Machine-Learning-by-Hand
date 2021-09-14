import numpy as np
import pandas as pd

class CARTNode:
    
    def __init__(self):
        self.cut_var = None
        self.cut_point = None
        self.avg = None
        self.depth = None
        self.num = None
        self.left = None
        self.right = None
        
class RegressionTree:
    
    def __init__(self,max_depth=float('inf'),min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def compute_loss(self,df_var,df_y,s):
        df = pd.concat([df_var,df_y],axis=1)
        df.columns = ['x','y']
        df_1 = df[df['x']<=s]
        df_2 = df[df['x']>s]
        c_1 = np.mean(df_1['y'])
        c_2 = np.mean(df_2['y'])
        loss = np.sum((df_1['y']-c_1).values**2)+np.sum((df_2['y']-c_2).values**2)
        return loss
    
    def fit(self,X,Y):
        self.var_num = X.shape[1]
        df = pd.concat([pd.DataFrame(X,columns = ['var'+str(i) for i in range(self.var_num)]),\
            pd.DataFrame(Y,columns = ['y'])],axis=1)
        self.root = CARTNode()
        self.build_tree(self.root,df)
    
    def build_tree(self,node,df,depth=0):
        node.avg = np.mean(df['y'])
        node.num = len(df)
        node.depth = depth
        
        if node.depth < self.max_depth and node.num > self.min_samples_leaf:
        
            ## 寻找切分变量和切分点
            cut = []
            for j in range(self.var_num):
                s_list = sorted(np.unique(df['var'+str(j)]))[:-1]
                for s in s_list:
                    loss = self.compute_loss(df[['var'+str(j)]],df[['y']],s)
                    cut.append([[j,s],loss])
        
            loss = [c[1] for c in cut]
            min_loss = min(loss)
            cut_var, cut_point = [c[0] for c in cut][loss.index(min_loss)]
        
            node.cut_var = 'var'+str(cut_var)
            node.cut_point = cut_point
        
            ## 递归
            node.left = CARTNode()
            self.build_tree(node.left,df[df[node.cut_var]<=node.cut_point],depth+1)
            node.right = CARTNode()
            self.build_tree(node.right,df[df[node.cut_var]>node.cut_point],depth+1)
            
    def print_node(self,node,var_dict,layer=0):
        if node.left is None and node.right is None:
            print('|'+'\t|'*layer+'---输出:'+str(round(node.avg,3))+',样本个数:'+str(node.num))
        
        if node.left is not None:
            print('|'+'\t|'*layer+'---'+var_dict[node.cut_var]+'<='+str(node.cut_point))
            self.print_node(node.left,var_dict,layer+1)
            
        if node.right is not None:
            print('|'+'\t|'*layer+'---'+var_dict[node.cut_var]+'>'+str(node.cut_point))
            self.print_node(node.right,var_dict,layer+1)
        

    def print_tree(self,col_name=None):
        var_name = ['var'+str(i) for i in range(self.var_num)]
        if col_name is not None:
            var_dict = dict(zip(var_name,col_name))
        else:
            var_dict = dict(zip(var_name,var_name))
        
        self.print_node(self.root,var_dict,layer=0)
            
    def predict_sample(self,new_X):
        node = self.root
        while node.cut_var is not None:
            if new_X[int(node.cut_var[3])]<=node.cut_point:
                node = node.left
            else:
                node = node.right
        return node.avg
        
    def predict(self,new_X):
        return np.apply_along_axis(self.predict_sample,axis=1,arr=new_X).reshape(-1,1)