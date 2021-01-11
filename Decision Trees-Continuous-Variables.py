#!/usr/bin/env python
# coding: utf-8

# In[81]:


get_ipython().run_line_magic('pylab', 'inline')
import scipy.stats
from collections import defaultdict  # default dictionary 
import copy


# In[82]:


class Node:
    def __init__(self,purity,klasslabel='',score=0,split=[],fidx=-1):
        self.lchild=None       
        self.rchild=None
        self.klasslabel=klasslabel        
        self.split=split
        self.score=score
        self.fidx=fidx
        self.purity=purity
        
        
    def set_childs(self,lchild,rchild):
        
        self.lchild=lchild
        self.rchild=rchild

        
    def isleaf(self):
        # Your Code Here
        if (self.lchild == None and self.rchild == None):
            return True
        else:
            return False
      #  print('returns true if the current node is leaf, else returns false')
    
    def isless_than_eq(self, X):
        if (X[self.fidx] < self.split):
            return True
        return False
        
    def get_str(self):        
        if self.isleaf():
            return 'C(class={},Purity={})'.format(self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split) 


# In[143]:


## Your code goes here...

class DecisionTree:
    ''' Implements the Decision Tree For Classification... '''
    def __init__(self, purityp, exthreshold,maxdepth=10,tree=None):        
        self.purity=purityp
        self.exthreshold=exthreshold
        self.maxdepth=maxdepth
        self.tree=tree
        
    def train(self, X, Y):
        ''' Train Decision Tree using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        nexamples,nfeatures=X.shape

        self.tree = self.build_tree(X,Y,self.maxdepth)

        
    def build_tree(self, X, Y, depth):
        """ 
            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...
        """
        nexamples, nfeatures=X.shape
        (values,counts) = np.unique(Y,return_counts=True)
        ind=np.argmax(counts)
        current_purity = counts[ind]/Y.shape[0]
        klasses,C=np.unique(Y,return_counts=True)
        # Base Cond.   
        if (depth == 0 or current_purity>=self.purity):
            (values,counts) = np.unique(Y,return_counts=True)
            ind=np.argmax(counts)
            clabel = values[ind]
            leaf = Node(current_purity, clabel)
            return leaf
        
        leftN = None
        rightN = None
        
        best_split = 0.0
        best_score = 0.0
        feature = -1
        leftIdx = None
        rightIdx = None

        for i in range (0, X.shape[1]):
            split,mingain,Xlidx,Xridx=self.evaluate_numerical_attribute(X[:,i],Y)
            if (mingain > best_score):
                best_score = mingain
                best_split = split
                feature = i
                leftIdx = Xlidx
                rightIdx = Xridx
        n = Node(purity=current_purity,klasslabel='', score=best_score, split=best_split, fidx=feature)
        n.lchild = self.build_tree(X[leftIdx], Y[leftIdx], depth-1)
        n.rchild = self.build_tree(X[rightIdx], Y[rightIdx], depth-1)
        return n
        
        
    def test(self, X):
        
        ''' Test the trained classifiers on the given set of examples 
        
                   
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''
        
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)
        return pclasses
        
    def calc_entropy(self, prob1, prob2, prob3):
        if (prob1 == 0):
            prob1 += 0.0000001
        if (prob2 == 0):
            prob2 += 0.0000001
        if (prob3 == 0):
            prob3 += 0.0000001
        return - (prob1*math.log(prob1,2) + prob2*math.log(prob2,2) + prob3*math.log(prob3,2))

    
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx]
        sY=Y[sidx]
        split = 0.0
        score = 100.0
        i_g = 0
        p_i = 0.0 
        p_j = 0.0 
        p_k = 0.0 
        
        for i in range (0, sY.shape[0]):
            if (sY[i] == 'Iris-setosa'):
                p_i += 1
            elif (sY[i] == 'Iris-versicolor'):
                p_j += 1
            else:
                p_k += 1
        entropy = self.calc_entropy(p_i/sY.shape[0], p_j/sY.shape[0], p_k/sY.shape[0])
        for j in range(0, sY.shape[0]):
            tsplit = f[j]
            count_greater = 0
            count_lesser = 0
            count_i = 0.0  
            count_j = 0.0 
            count_k = 0.0  

            count_l = 0.0 
            count_m = 0.0 
            count_n = 0.0 
            for i in range (0, sY.shape[0]):
                if f[i] >= tsplit:
                    count_greater += 1
                    if (sY[i] == 'Iris-setosa'):
                        count_i += 1
                    elif (sY[i] == 'Iris-versicolor'):
                        count_j += 1
                    else:
                        count_k += 1
                else:
                    count_lesser += 1
                    if (sY[i] == 'Iris-setosa'):
                        count_l += 1
                    elif (sY[i] == 'Iris-versicolor'):
                        count_m += 1
                    else:
                        count_n += 1
            if (count_greater == 0 or count_lesser == 0):
                continue
            greater_entropy = self.calc_entropy(count_i/count_greater, count_j/count_greater, count_k/count_greater)
            lesser_entropy = self.calc_entropy(count_l/count_lesser, count_m/count_lesser, count_n/count_lesser)
            entropy_split = (count_greater/sY.shape[0])*greater_entropy + (count_lesser/sY.shape[0])*lesser_entropy
            information_gain = entropy - entropy_split
            if (entropy_split < score):
                score = entropy_split
                i_g = information_gain
                split = tsplit
        score = i_g
        leftChildInd = np.where(f <  split)[0]
        RightChildInd = np.where(f >= split)[0]
        
        return split, score, leftChildInd, RightChildInd          
    
    def predict(self, X):
        
        """
        Test the trained classifiers on the given example X
        
                   
            Input:
            ------
            X: [1 x d] a d-dimensional test example.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        pclass = []
        for i in range (0, X.shape[0]):
            temp = self._predict(self.tree, X[i,:])
            pclass.append(temp)
        return pclass
    
    def _predict(self,node, X):
        # YOUR CODE HERE
        if (node.isleaf() == True):
            temp = node.klasslabel
            return temp
        else:
            if (node.isless_than_eq(X) == True):
                return self._predict(node.lchild, X)
            else:
                return self._predict(node.rchild, X)

    def __str__(self):
        
        return self.__print(self.tree)        
        
     
    def find_depth(self):
        
        return self._find_depth(self.tree)
    
    
    def _find_depth(self,node):
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild),self._find_depth(node.rchild))+1
        
    def __print(self,node,depth=0):
        
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild,depth+1)

        
        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild,depth+1)
        
        return ret


# In[144]:


import pandas as pd
import numpy as np
import tools as t # set of tools for plotting, data splitting, etc..


# In[145]:


#load the data set
data=pd.read_csv('./iris.data')
data.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']
print (data.describe())


# In[146]:


# Get your data in matrix (X ,Y)
temp1 = data[['SepalLength','SepalWidth','PetalLength','PetalWidth']].dropna()
temp2 = data['Class'].dropna()
X = np.asarray(temp1)
Y = np.asarray(temp2)
    
print (" Data Set Dimensions=", X.shape, " True Class labels dimensions", Y.shape)   


# In[147]:


print (len(Y))
feat=[0,1]
dt=DecisionTree(0.95,5,2)
feat=[0,1]
dt.classes=np.unique(Y)
dt.nclasses=len(np.unique(Y))
split,mingain,Xlidx,Xridx=dt.evaluate_numerical_attribute(X[:,0],Y)
print (split)


# In[148]:


print (len(Y))
dt=DecisionTree(0.95,5)
dt.train(X[:,feat],Y)
g,s,xl,xr=dt.evaluate_numerical_attribute(X[:,2],Y)
print (g, s, xl, xr)


# In[149]:


# %pdb
print (" Plotting the Decision Surface of Training Set... ")
t.plot_decision_regions(X[:,feat],Y,clf=dt, res=0.1, cycle_marker=True, legend=1)


# In[150]:


# Split your data into training and test-set... 
# see the documentation of split_data in tools for further information...
Xtrain,Ytrain,Xtest,Ytest=t.split_data(X,Y)

print (" Training Data Set Dimensions=", Xtrain.shape, "Training True Class labels dimensions", Ytrain.shape)
print (" Test Data Set Dimensions=", Xtest.shape, "Test True Class labels dimensions", Ytrain.shape)   


# In[151]:


# Lets train a Decision Tree Classifier on Petal Length and Width
feat=[0,1]
dt=DecisionTree(0.95,5)
dt.train(Xtrain[:,feat],Ytrain)


# In[152]:


#Lets test it on the set of unseen examples...
pclasses=dt.predict(Xtest[:,feat])


# # Let's See How Good we are doing

# In[153]:


#Lets see how good we are doing, by finding the accuracy on the test set..
print (np.sum(pclasses==Ytest))
print ("Accuracy = ", np.sum(pclasses==Ytest)/float(Ytest.shape[0]))


# # Lets Train on All 4 Features and all 3 classes 

# In[154]:


# Split your data into training and test-set... 
# see the documentation of split_data in tools for further information...
Xtrain,Ytrain,Xtest,Ytest=t.split_data(X,Y)

print (" Training Data Set Dimensions=", Xtrain.shape, "Training True Class labels dimensions", Ytrain.shape)
print (" Test Data Set Dimensions=", Xtest.shape, "Test True Class labels dimensions", Ytrain.shape)   


# In[155]:


dt=DecisionTree(0.95,5)
dt.train(Xtrain,Ytrain)
pclasses=dt.predict(Xtest)
#Lets see how good we are doing, by finding the accuracy on the test set..
print (np.sum(pclasses==Ytest))
print ("Accuracy = ", np.sum(pclasses==Ytest)/float(Ytest.shape[0]))


# In[ ]:





# In[ ]:




