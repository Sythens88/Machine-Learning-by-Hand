import pandas as pd
import numpy as np

class CARTNode:
    
    def __init__(self):
        self.cut_var = None
        self.cut_point = None
        self.avg = None
        self.depth = None
        self.num = None
        self.left = None
        self.right = None
        
        
class ClassificationTree:
    
    def __init__(self,max_depth=float('inf'),min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def compute_loss(self,df_var,df_y,s):
        df = pd.concat([df_var,df_y],axis=1)
        df.columns = ['x','y']
        df_1 = df[df['x']<=s]
        df_2 = df[df['x']>s]
        num_1 = df_1['y'].value_counts().values
        num_1 = num_1/np.sum(num_1)
        gini_1 = 1-np.sum(num_1**2)
        num_2 = df_2['y'].value_counts().values
        num_2 = num_2/np.sum(num_2)
        gini_2= 1-np.sum(num_2**2)
        gini = len(df_1)/(len(df_1)+len(df_2))*gini_1+len(df_2)/(len(df_1)+len(df_2))*gini_2
        return gini
    
    def fit(self,X,Y,col_num=None):
        self.var_num = X.shape[1]
        col_num = X.shape[1] if col_num is None else col_num
        df = pd.concat([pd.DataFrame(X,columns = ['var'+str(i) for i in range(self.var_num)]),\
            pd.DataFrame(Y,columns = ['y'])],axis=1)
        self.root = CARTNode()
        self.build_tree(self.root,df,col_num)
    
    def build_tree(self,node,df,col_num,depth=0):
        node.avg = df['y'].value_counts().idxmax()
        node.num = len(df)
        node.depth = depth
        
        if node.depth < self.max_depth and node.num > self.min_samples_leaf and\
            len(df['y'].value_counts())!=1:
        
            ## 寻找切分变量和切分点
            cut = []
            idx_list = list(np.random.permutation([i for i in range(self.var_num)]))[:col_num]
            for j in range(self.var_num):
                if j in idx_list:
                    s_list = sorted(np.unique(df['var'+str(j)]))[:-1]
                    for s in s_list:
                        loss = self.compute_loss(df[['var'+str(j)]],df[['y']],s)
                        cut.append([[j,s],loss])
        
            loss = [c[1] for c in cut]
            if len(loss)!=0:
                
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