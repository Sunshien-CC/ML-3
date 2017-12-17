import pickle
import numpy as np
from sklearn.metrics import classification_report
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.clf=[]#用来保存分类器的数组
        self.alpha=[]#用来保存α值的数组
    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        weight=np.ones(X.shape[0])/X.shape[0]#初始化样本权重，每个样本权重为 1/n
        for i in range(self.n_weakers_limit):
            print('正在训练第',i,'个分类器')
            model=self.weak_classifier(max_depth=1).fit(X,y,sample_weight=weight)#第i个分类器
            self.clf.append(model)#将第i个分类器保存到数组
            #计算第i个分类器的错误率
            error=0.0
            for j in range(X.shape[0]):
                if self.clf[i].predict(X[j].reshape(1,-1))!=y[j]:
                    error+=weight[j]#在weight分布下的错误率
            if error>0.5:
                break
            #计算α值
            alpha_i=0.5*np.log((1-error)/error)
            self.alpha.append(alpha_i)#保存α到数组中
            #更新样本权重       
            for k in range(X.shape[0]):
                weight[k]=weight[k]*np.exp(-y[k]*self.alpha[i]*self.clf[i].predict(X[k].reshape(1,-1)))
            #除以规范化因子
            Z_i=sum(weight)#规范化因子
            weight=np.array([w/Z_i for w in weight])

    def predict_scores(self, X,y):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_pre=self.predict(X)
        with open('report.txt',mode='w') as f:
            f.write(classification_report(y_pre,y))
        

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        result=np.zeros(X.shape[0])
        for i in range(len(self.alpha)):#每个分类器预测出来的值乘以α再相加
            result +=self.alpha[i]*self.clf[i].predict(X)
        #符号函数，将值映射到+1和-1
        for i in range(len(result)):
            if result[i]>0.0:
                result[i]=1.0
            else:
                result[i]=-1.0
        return result
    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
