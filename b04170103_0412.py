
# coding: utf-8

# In[1]:


x,y,z=3,4,5
print(x)
print(y)
print(z)


# In[12]:


#交換
x,y=3,4
x,y=y,x
print(x)
print(y)


# In[5]:


x,y,z=3,4,5
x+=1 #x=x+1
y*=2 #y=y*2
z**=3 #**(次方)
print(x,y,z)


# In[16]:


x=1
y=10*x
x=x+y
print(x)
print(y)


# In[9]:


#海龍公式
import math
a,b,c=3,4,5
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))
print(area)
import math
a,b,c=12,33,25
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))
print(area)


# In[3]:


x=254
print(type(x))
x="write"
print(type(x))
x=254.0
print(type(x))
x=True
print(type(x))


# In[7]:


#2,8,16進字
print(0b111)
print(0o137)
print(0xff)


# In[10]:


import math
print(4*(math.pi*4.5*4.5*4.5)/3)


# In[14]:


x=3.141592627
print(x-3.14)
print(2.1-2.0)


# In[29]:


#匯圖
import matplotlib.pyplot as pt
x=[1,2,3,5,8]
y=[2,5,7,8,6]
z=[3,5,7,2,9]
pt.plot(x,y,"--",color="green",label="October")
pt.plot(y,z,"^",label="August")
pt.legend() #顯示小圖
pt.show()


# In[34]:


import matplotlib.pyplot as pt
x=[1,5,4,8,7,3]
y=[2,7,6,4,3,5]
pt.bar(x,y,label="December")
pt.legend()
pt.show()


# In[38]:


import matplotlib.pyplot as pt
x=[1,5,7,9,5]
y=[4,8,5,6,3]
pt.scatter(x,y)
pt.show()


# In[44]:


import numpy as np
import matplotlib.pyplot as pt
x=np.random.random(5000)
y=np.random.random(5000)
pt.scatter(x,y)
pt.show()


# In[63]:


import numpy as np
import matplotlib.pyplot as pt
x=np.arange(0,360)
y=np.sin(x*np.pi/180)
z=np.cos(x*np.pi/180)
pt.xlim(0,360)
pt.ylim(-1.2,1.2)
pt.title("Sin & Cos Vave")
pt.xlabel("Degree")
pt.ylabel("Value")
pt.plot(x,y,label="Sin")
pt.plot(x,z,label="Cos")
pt.legend()
pt.show()


# In[16]:


from sklearn import datasets,cluster,metrics
import matplotlib.pyplot as pt
iris=datasets.load_iris()
silhouette_avgs=[]
#print(iris["DESCR"])
#print(iris["data"])
#print(iris["target"])
lkk=range(2,10)
for k in lkk:
    iris_km=cluster.KMeans(n_clusters=k).fit(iris["data"])
    #print(iris_km.labels_)
    silhouette_avg=metrics.silhouette_score(iris["data"],iris_km.labels_)
    #print(silhouette_avg)
    silhouette_avgs.append(silhouette_avg)
pt.bar(lkk,silhouette_avgs)
pt.show()
    


# In[21]:


from sklearn import datasets
import matplotlib.pyplot as pt
digits=datasets.load_digits()
print(digits["DESCR"])
print(digits["data"])
print(digits["target"])
pt.figure(1,figsize=(3,3))
pt.imshow(digits.images[0],cmap=pt.cm.gray_r,interpolation='nearrest')
pt.show()


# In[1]:


from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt
boston=datasets.load_boston()
#print(boston.DESCR)
#print(boston.target)
print(boston.data)
#CRIM(犯罪率) ZN(房星大於25000ft比率)
#INDOUS(住宅比率) CHAS(有吳臨河) NOX(空汙比率) RM(房間數)
#AGE(自有住宅比例) DIS(離市中心距離) RAD(離高速公路距離)
#TAX(房屋稅率) PTRATIO(小學老師比率) B(黑人比率)
#STAT(低收人比率) MEDV(受僱者收入)4
lr=linear_model.LinearRegression()
predict=cross_val_predict(lr,boston.data,boston.target,cv=10)
plt.figure()
plt.scatter(boston.target,predict)
y=boston.target
plt.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
plt.plot()
plt.show()
print(predict)


# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
data=datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
#plt.imshow(data.images[0],cmap='gray',interpolation='nearest')
#plt.show()
#把影像變成一列
targets=data.target
data=data.images.reshape(len(data.images),-1)
#訓練資料30張臉(300張圖片)，測試資料10張臉(100張圖片)
train=data[targets<30]
test=data[targets>=30]
# 從100張測試影像中,亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
from sklearn.utils import check_random_state
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分
#， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
#決定預測的演算法
from sklearn.linear_model import LinearRegression
ESTIMATORS = {
    "Linear regression": LinearRegression(),
}
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train) #模型訓練
    y_test_predict[name] = estimator.predict(X_test) 
    #模型預測
# Plot the completed faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)
for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)
        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
plt.show()

from sklearn import datasets
from sklearn.utils import check_random_state
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
targets = data.target
data = data.images.reshape((len(data.images), -1)) #把影像變成一列
train = data[targets < 30]
test = data[targets >= 30]
# 測試影像從100張亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

ESTIMATORS = {
    "Linear regression": LinearRegression(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")

    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()

