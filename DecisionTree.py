import numpy as np
from collections import Counter
class Node:
    def __init__(self,feature=None,threshold=None, left=None, right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    def is_leaf_node(self):
        return self.value is not None
class DecisionTree:
    def __init__(self,min_sample_split=2,max_depth=100,num_feature=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.num_feature=num_feature
        self.root=None
    def _grow_tree(self,X,y,depth=0):
        m,n=X.shape
        n_lable=len(np.unique(y))
        #stop criteria
        #stop when dep>max_dep; label_ 100% 1 class ;num of examples< threshold(min_sample)=>avoid overfiting
        if(depth>=self.max_depth or n_lable==1  or m<self.min_sample_split):
            leaf_val=self._most_label(y)
            return Node(value=leaf_val)
        
        
        #find best split
        feature_indxs=np.random.choice(n,self.num_feature,replace=False)# group ramdom feature
        best_feature,best_threshold=self._best_split(X,y,feature_indxs)

        #child nodes
        left_,right_=self._split_dataset(X,best_threshold,best_feature)
        left=self._grow_tree(X[left_, :],y[left_],depth+1) #X[left_, :] lấy toàn bộ bên phải + cột(feature) 
        right=self._grow_tree(X[right_, :],y[right_],depth+1)
        return Node(best_feature,best_threshold,left,right)    
    
    def _best_split(self,X,y,feature_indxs):
        best_feature = -1
        max_info_gain,threshold = -1,None
        for fea in feature_indxs:
            X_colum=X[ : , fea]# lấy riêng 1 cột feature
            values=np.unique(X_colum)# get only, sort
            thresholds = (values[:-1] + values[1:]) / 2
            for thre in thresholds:
                gain=self._compute_info_gain(X,y,fea,thre)
                if gain>max_info_gain:
                    max_info_gain=gain
                    threshold=thre
                    best_feature=fea
        return best_feature,threshold
    
    
    #entropy
    def _entropy(self, y):
        entropy = 0
        if len(y) == 0:
            return 0
        entropy = sum(y[y==1])/len(y)
        if entropy == 0 or entropy == 1:
            return 0
        else:
            return -entropy*np.log2(entropy) - (1-entropy)*np.log2(1-entropy)


         
    #split data to r,l nỏte
    def _split_dataset(self, X, threshold, feature):
        left_indices = []
        right_indices = []
        for i,x in enumerate(X):
            if x[feature] <=threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        return left_indices, right_indices  
     
    def _compute_info_gain(self,X,y,feature,threshold):
        left_indices,right_indices=self._split_dataset(X,threshold,feature)
        node_entro=self._entropy(y)
        
        w_left = len(left_indices)/len(X)
        w_right = len(right_indices)/len(X)
        p_left = self._entropy(y[left_indices])
        p_right = self._entropy(y[right_indices])
        
        weighted_entropy = w_left * p_left + w_right * p_right
        return node_entro-weighted_entropy
        
    def _most_label(self,y):
        counter=Counter(y)
        return counter.most_common(1)[0][0]
        
    def fit(self, X,y):
        if self.num_feature is None:
            self.num_feature = X.shape[1]
        else:
            self.num_feature = min(self.num_feature, X.shape[1])
        self.root=self._grow_tree(X,y)
            
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)