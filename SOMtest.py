
# coding: utf-8

# In[6]:


import numpy as np

from SOM import SOM




        #self.show("final.jpg")



# In[7]:





# In[9]:

t_set = np.random.randint(256, size=(15, 3))
s = SOM(200, 200, 3, 100, 0.1)
#s.setWeight(w)
s.show("before.jpg")
# t_set = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200], [120, 0, 100]])
#t_set = np.random.randint(256, size=(15, 3))
s.teach(t_set)
s.show("ref.jpg")


# In[10]:

from SVDinitializeWeight import initializeWeight2DVector
s2 = SOM(200, 200, 3, 100, 0.1)
w = initializeWeight2DVector(t_set,200,200)
s2.setWeight(w)
s2.teach(t_set)
s2.show("final.jpg")


# In[ ]:

from PCAinitializeWeight import pcainitializeWeight2DVector
s2 = SOM(200, 200, 3, 100, 0.1)
w = pcainitializeWeight2DVector(t_set,200,200)
s2.setWeight(w)
s2.teach(t_set)
s2.show("final2.jpg")


