from PIL import Image
from feature import NPDFeature
from ensemble import AdaBoostClassifier
import ensemble
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,classification_report

if __name__ == "__main__":
    # write your code here
    dir='./datasets/original/'
    face=[]
    nonface=[]
    #读取图片
    for i in range(500):
        face_im=Image.open(dir+'face/face_'+"%03d" % i+'.jpg')
        nonface_im=Image.open(dir+'nonface/nonface_'+'%03d'%i+'.jpg')
        face_im=face_im.convert('L')#灰度化
        face_im=face_im.resize((24,24)) #缩小尺寸
        nonface_im=nonface_im.convert('L')
        nonface_im=nonface_im.resize((24,24)) 
        face.append(np.array(face_im))#转为ndarray
        nonface.append(np.array(nonface_im))
    feature_face=[]
    feature_nonface=[]
    for i in range(500) :
        feature_face.append(NPDFeature(face[i]).extract())
        feature_nonface.append(NPDFeature(nonface[i]).extract())
    # #缓存特征
    # AdaBoostClassifier.save(feature_face,'feature_face')
    # AdaBoostClassifier.save(feature_nonface,'feature_nonface')
    # #读取缓存的特征
    # feature_face=np.array(AdaBoostClassifier.load('feature_face'))
    # feature_nonface=np.array(AdaBoostClassifier.load('feature_nonface'))

    data=np.row_stack((feature_face,feature_nonface))
    label=np.concatenate((np.ones(500),-np.ones(500)))
    X_train,X_validation,y_train,y_validation=train_test_split(data,label,test_size=0.3,random_state=1000)
    #Adaboost 20个分类器，每个决策树只有一个节点
    model=AdaBoostClassifier(DecisionTreeClassifier,20)
    model.fit(X_train,y_train)
    model.predict_scores(X_validation,y_validation)

    #单个分类器
    model=DecisionTreeClassifier(max_depth=1).fit(X_train,y_train)
    y_pre=model.predict(X_validation)
    print(classification_report(y_pre,y_validation))