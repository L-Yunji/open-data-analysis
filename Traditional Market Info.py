#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns

#구버젼의 주피터 노트북에서 그래프가 보이는 설정
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# 한글폰트 설정
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# 폰트가 선명하게 보이도록 retina 설정
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")


# In[4]:


# 한글폰트와 마이너스 폰트 설정 확인

pd.Series([-1, 0, 1, 3, 5]).plot(title="한글폰트")


# # 데이터 불러오기

# In[7]:


df = pd.read_csv('data/전통시장현황.csv', encoding='CP949')
df.shape


# In[8]:


df.head()


# In[9]:


df.info()


# In[12]:


# 결측치 보기
df.isnull().sum().plot.barh(figsize=(10,25))


# ### 사용하지 않는 컬럼 제거하기

# In[15]:


# drop을 하는 방법도 있지만 사용할 컬럼만 따로 모아서 보는 방법도 있습니다.
# 여기에서는 사용할 컬럼만 따로 모아서 사용합니다.
df.columns 


# In[19]:


columns = ['시장명','시군구','시도','보유갯수 - 16시장전용 고객주차장',
           '시장/상점가의 주력 상품 여부(1=있음, 2=없음)','보유현황 - 10쇼핑카트(1=있음, 2=없음)',
           '시장/상점가의 주력 상품의 상품명']


# In[20]:


df = df[columns].copy()
df.shape


# In[21]:


df.info()


# In[33]:


# 시도가 서울로 시작하는 데이터만 봅니다.
# 또, df_seoul 이라는 변수에 결과를 저장합니다.
# 새로운 변수에 데이터프레임을 할당할 때 copy()를 사용하는 것을 권장합니다.
df_seoul = df[df["시도"] == "서울"].copy()
df_seoul.shape


# ### 일부 텍스트가 들어가는 데이터만 가져오기
# * 시장명에서 '시장'이 들어가있는 시장만 추출합니다.

# In[34]:


# 시장만 가져오기
df_seoul[df_seoul["시장명"].str.contains("시장")]


# In[36]:


df_seoul.loc[df_seoul["시장명"].str.contains("시장"), "시장명"].shape


# In[ ]:




