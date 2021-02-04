#!/usr/bin/env python
# coding: utf-8

# # 실습 파일
# * https://colab.research.google.com/github/corazzon/open-data-analysis-basic/blob/master/01-apt-price-input.ipynb
# # 결과 파일
# * https://colab.research.google.com/github/corazzon/open-data-analysis-basic/blob/master/01-apt-price-output.ipynb

# In[1]:


import pandas as pd


# In[2]:


df_last = pd.read_csv("data/주택도시보증공사_전국 평균 분양가격(2019년 12월).csv", encoding="cp949")
df_last.shape


# In[3]:


df_last.head() #위에 데이터 5개 불러오기


# In[4]:


df_last.tail() #마지막 데이터 5개 불러오기 NaN : 결측치


# In[5]:


get_ipython().run_line_magic('ls', 'data')


# In[6]:


df_first = pd.read_csv("data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_first.shape


# In[7]:


df_first.head()


# In[8]:


df_first.tail()


# In[9]:


df_last.info()


# In[10]:


True == 1


# In[11]:


False == 0


# In[12]:


True + True + False


# In[13]:


df_last.isnull() #결측치 수 구하기


# In[14]:


df_last.isnull().sum() #결측치 합 구하기


# # 데이터 타입 바꾸기

# In[15]:


df_last["분양가격(㎡)"] #숫자데이터인데 object > 수치데이터로 바꿔야 함.


# In[16]:


import numpy as np


# In[17]:


np.nan


# In[18]:


df_last["분양가격"] = pd.to_numeric(df_last["분양가격(㎡)"], errors='coerce')
df_last["분양가격"].head(1)


# # 평당 분양가격

# In[19]:


df_last["평당분양가격"] = df_last["분양가격"]*3.3 #가격비교 위해 단위 맞춰준다.
df_last.head(1)


# In[20]:


df_last.info()


# In[21]:


df_last["분양가격(㎡)"].describe() #object타입일 때 요약


# In[22]:


df_last["분양가격"].describe() 


# # 분양가격(m^2)
# * object > 숫자 데이터
# * 수치데이터 일때 요약 count값이 달라지는 이유는 공백 데이터 때문!
# * unique : 중복되지 않은 값
# * top : 빈번하게 등장하는 수
# * freq : 가장 빈번하게 등장하는 수가 몇번 등장하는 지
# 
# # 분양가격
# * mean : 평균값
# * std : 표준편차
# * min : 최소값
# * 25% : 일사분위수

# # 규모구분을 전용면적으로 바꿔 전처리 (초과, 이하, 공백 처리)

# In[23]:


df_last["규모구분"].unique() #규모구분을 전용면적으로 바꾸자! (반복되므로)


# In[24]:


df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적", "")
df_last["전용면적"] = df_last["전용면적"].str.replace("초과", "~")
df_last["전용면적"] = df_last["전용면적"].str.replace("이하", "")
df_last["전용면적"] = df_last["전용면적"].str.replace(" ","").str.strip() #공백제거
df_last["전용면적"]


# In[25]:


df_last.info()


# In[26]:


df_last = df_last.drop(["규모구분", "분양가격(㎡)"], axis=1)


# In[27]:


df_last.head(1)


# # groupby로 데이터 집계하기
# * df.groupby(["인덱스로 사용할 컬럼명"])["계산할 컬럼 값"].연산()

# In[28]:


df_last.groupby(["지역명"])["평당분양가격"].mean() 


# In[29]:


df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[30]:


#지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
df_last.groupby(["지역명", "전용면적"])["평당분양가격"].mean()


# In[31]:


#unstack() 끝에있는 값이 컬럼 값으로 오게 된다.
#round() 소수점 제거
df_last.groupby(["전용면적","지역명"])["평당분양가격"].mean().unstack().round()


# In[32]:


#연도, 지역명으로 평당분양가격의 평균 구함
df_last.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().round()


# In[33]:


#.T 행과열을 바꾸고 싶다.
df_last.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().round().T


# In[34]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.

g = df_last.groupby(["연도","지역명"])["평당분양가격"].mean()
g
# g.unstack().transpose()


# # pivot table로 데이터 집계하기
# * groupby로 했던 작업을 pivot_table로 똑같이 집계합니다.
# * data frame형태로 나타남
# * groupby는 빠르다
# * pivot_table은 직관적으로 알기 쉽다.

# In[35]:


pd.pivot_table(df_last, index =["지역명"], values=["평당분양가격"], aggfunc="mean")


# In[36]:


# df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()
#pivot은 연산을 하지 않는다. 데이터 형태만 바꾼다.
pd.pivot_table(df_last, index="전용면적", values="평당분양가격")


# In[37]:


#지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
#df_last.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().round()
df_last.pivot_table(index="전용면적",columns= "지역명", values="평당분양가격").round()


# In[38]:


#g = df_last.groupby(["연도","지역명"])["평당분양가격"].mean()
p = pd.pivot_table(df_last, index=["연도","지역명"], values="평당분양가격")
p.loc[2018]


# # 데이터 시각화

# In[39]:


import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")


# In[40]:


#rot =지역명 바로 세우기 figsize =그래프 크기 sort_values() =정렬 sort_values(ascending=False) =큰순서대로 정렬
g = df_last.groupby(["지역명"])["평당분양가격"].mean().sort_values(ascending=False)
g.plot.bar(rot=0, figsize=(10,3))


# In[41]:


# 전용면적으로 분양가격의 평균을 구하고 막대그래프로 시각화합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean().plot.bar()


# In[42]:


#연도별 분양가격의 평균을 구하고 막대그래프로 시각화 합니다.
df_last.groupby(["연도"])["평당분양가격"].mean().plot()


# In[43]:


df_last.pivot_table(index="월", columns="연도", values="평당분양가격").plot.box()


# In[44]:


p = df_last.pivot_table(index="월", columns=["연도","전용면적"], values="평당분양가격")
p.plot.box(figsize = (15,3), rot=30)


# In[45]:


p = df_last.pivot_table(index="연도", columns=["지역명"], values="평당분양가격")
p.plot(figsize = (15,3), rot=30)


# # Seaborn으로 시각화 해보기
# * 연산을 하지 않아도 그래프 그릴 수 있다
# * subplot그리기 쉽다

# In[46]:


import seaborn as sns


# In[47]:


#ci = "sd" >표준편차 None > 그리지 않음
plt.figure(figsize=(10,3))
sns.barplot(data=df_last, x="지역명", y="평당분양가격")


# In[48]:


sns.barplot(data=df_last,x="연도", y="평당분양가격")


# In[49]:


#hue 옵션을 통해 지역별로 다르게 표시해 봅니다.
#legend값이 떨어져서 보이게 하는 것 plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(10,5))
sns.lineplot(data=df_last, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# * relplot

# In[50]:


sns.relplot(data=df_last, x="연도", y="평당분양가격", hue="지역명", kind="line", col="지역명", col_wrap=4, ci=None)


# In[51]:


sns.catplot(data=df_last, x = "연도", y="평당분양가격", kind="bar", col="지역명", col_wrap=4)


# * boxplot과 violinplot

# In[52]:


sns.boxplot(data=df_last, x = "연도", y ="평당분양가격")


# In[53]:


#hue옵션을 주어 전용면적별로 다르게 표시해 봅니다.
plt.figure(figsize=(12,3))
sns.boxplot(data=df_last, x = "연도", y ="평당분양가격", hue ="전용면적")


# In[54]:


#박스 안 데이터 분포를 알 수 있다

sns.violinplot(data=df_last, x = "연도", y ="평당분양가격")


# ### lmplot과 swarmplot

# In[102]:


# 연도별 평당분양가격을 lmplot으로 그려봅니다.
# hue 옵션으로 전용면적을 표현해 봅니다.
# scatterplot의 회귀선을 그려준다. > regplot
sns.regplot(data=df_last, x="연도", y="평당분양가격")


# In[56]:


sns.lmplot(data=df_last, x="연도", y="평당분양가격", hue="전용면적", col="전용면적", col_wrap=3)


# In[57]:


#swarmplot은 범주형(카테고리)데이터의 산점도를 표현하기에 적합합니다.
#plt.figure(figsize=(15,3))
#sns.swarmplot(data=df_last, x= "연도", y="평당분양가격")


# ### 이상치보기

# In[59]:


df_last["평당분양가격"].describe()


# In[60]:


max_price = df_last["평당분양가격"].max()
max_price


# In[61]:


# 서울의 평당분양가격이 특히 높은 데이터가 있습니다. 해당 데이터를 가져옵니다.
df_last[df_last["평당분양가격"] == max_price]


# ### 수치데이터 히스토그램 그리기
# displot은 결측치가 있으면 그래프를 그릴 때 오류가 납니다. 따라서 결측치가 아닌 데이터만 따로 모아서 평당분양가격을 시각화하기 위한 데이터를 만듭니다. 데이터프레임의 .loc를 활용하여 겨ㄹ측치가 없는 데이터에서 평당분양가격만 가져옵니다.

# In[62]:


# .loc[행]
# .loc[행, 열]
# bins > 몇개의 막대로 표현할 것인지
h = df_last["평당분양가격"].hist(bins=20)


# In[63]:


h = df_last.hist()


# In[64]:


# .loc[행]
# .loc[행, 열]
# bins > 몇개의 막대로 표현할 것인지
price = df_last.loc[df_last["평당분양가격"].notnull(), "평당분양가격"]


# In[65]:


#distplot으로 평당분양가격을 표현해 봅니다.

sns.distplot(price)


# * displot을 산마루 형태의 ridge plot으로 그리기
# * https://seaborn.pydata.org/tutorial/axis_grids.html#conditional-small-multiples
# * https://seaborn.pydata.org/examples/kde_ridgeplot.html

# In[66]:


g = sns.FacetGrid(df_last, row="지역명",
                  height=1.7, aspect=4,)
g.map(sns.distplot, "평당분양가격", hist=False, rug=True)


# In[67]:


# sns.distplot(price, hist=False, rug=True)
#cumulative : 누적 그래프 kdeplot:커널 밀도 그래프
sns.kdeplot(price, cumulative=True)


# In[68]:


#pairplot
#.loc 행, 렬 함께 가져 올 수 있다
df_last_notnull = df_last.loc[df_last["평당분양가격"].notnull(), 
                          ["연도","월","평당분양가격", "지역명","전용면적"]]
sns.pairplot(df_last_notnull, hue="전용면적")


# In[69]:


# 규모구분(전용면적)별로  value_counts를 사용해서 데이터를 집계해 봅니다.
df_last["전용면적"].value_counts()


# # 1.8 2015년 8월 이전 데이터 보기

# In[70]:


pd.options.display.max_columns = 100


# In[71]:


df_last.head()


# In[72]:


df_first.info()


# In[73]:


df_first.head()


# In[74]:


#결측치가 있는지 봅니다.
df_first.isnull().sum()


# ### 1.8.1 melt로 Tidy data만들기
# pandas의 melt를 사용하면 데이터의 형태를 변경할 수 있습니다. df_first 변수에 담긴 데이터프레임은 df_last에 담겨있는 데이터프레임의 모습과 다릅니다. 같은 형태로 만들어주어야 데이터를 합칠 수 있습니다. 데이터를 병합하기 위해 melt를 사용해 열에 있는 데이터를 행으로 녹여봅니다.
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-by-melt
# * https://vita.had.co.nz/papers/tidy-data.pdf

# In[75]:


# pd.melt를 사용하며, 녹인 데이터는 df_first_melt 변수에 담습니다.
# col에 데이터 들어있으면 데이터 처리가 쉽지 않다.

df_first_melt = df_first.melt(id_vars="지역", var_name="기간", value_name="평당분양가격")
df_first_melt.head()


# In[76]:


df_first_melt.columns = ["지역명", "기간", "평당분양가격"]
df_first_melt.head(1)


# ### 1.8.2 연도와 월을 분리하기
# * pandas 의 string-handling 사용하기 : https://pandas.pydata.org/pandasdocs/stable/reference/series.html#string-handling

# In[77]:


date = "2013년12월"
date


# In[78]:


date.split("년")[0]


# In[79]:


date.split("년")[-1].replace("월", "")


# In[80]:


# parse+year라는 함수를 만듭니다.
# 연도만 반환하도록 하며, 반환하는 데이터는 int타입이 되도록 합니다.

def parse_year(date):
    year = date.split("년")[0]
    year = int(year)
    return year

y = parse_year(date)
print(type(y))
y


# In[81]:


parse_year(date)


# In[82]:


# parse_month 라는 함수를 만듭니다.
def parse_month(date):
    month = date.split("년")[-1].replace("월", "")
    month = int(month)
    return month


# In[83]:


parse_month(date)


# In[84]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 연도만 추출해서 새로운 컬럼에 담습니다.
df_first_melt["연도"] = df_first_melt["기간"].apply(parse_year)
df_first_melt.head(1)


# In[85]:


# df_first_melt 변수에 담긴 데이터프레임에서 
# apply를 활용해 월만 추출해서 새로운 컬럼에 담습니다.

df_first_melt["월"] = df_first_melt["기간"].apply(parse_month)
df_first_melt.head(1)


# In[86]:


# df_last와 병합을 하기 위해서는 컬럼의 이름이 같아야 합니다.
# sample을 활용해서 데이터를 미리보기 합니다.
df_last.sample()


# In[87]:


# 버전에 따라 tolist() 로 동작하기도 합니다.
# to_list() 가 동작하지 않는다면 tolist() 로 해보세요.
df_last.columns.to_list()


# In[92]:


df_last.columns.to_list()


# In[94]:


cols = ['지역명', '연도', '월', '평당분양가격']
cols


# In[97]:


# 최근 데이터가 담긴 df_last 에는 전용면적이 있습니다. 
# 이전 데이터에는 전용면적이 없기 때문에 "전체"만 사용하도록 합니다.
# loc를 사용해서 전체에 해당하는 면적만 copy로 복사해서 df_last_prepare 변수에 담습니다.
df_last_prepare = df_last.loc[
    df_last["전용면적"] == "전체", cols].copy()
df_last_prepare.head(1)


# In[98]:


# df_first_melt에서 공통된 컬럼만 가져온 뒤
# copy로 복사해서 df_first_prepare 변수에 담습니다.
df_first_prepare = df_first_melt[cols].copy()
df_first_prepare.head(1)


# ### 1.8.3 concat으로 데이터 합치기
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

# In[99]:


# df_first_prepare 와 df_last_prepare 를 합쳐줍니다.

df = pd.concat([df_first_prepare, df_last_prepare])
df.shape


# In[100]:


# 연도별로 데이터가 몇개씩 있는지 value_counts를 통해 세어봅니다.

df["연도"].value_counts(sort=False)


# ### 1.8.4 pivot_table 사용하기
# * https://pandas.pydata.org/docs/user_guide/reshaping.html#reshaping-and-pivot-tables

# In[106]:


t = pd.pivot_table(df, index="연도", columns="지역명", values="평당분양가격").round()
t


# In[112]:


# 위에서 그린 피봇테이블을 해트맵으로 표현해 봅니다.
plt.figure(figsize=(15,7))
sns.heatmap(t, cmap="Blues", annot=True, fmt=".0f")


# In[113]:


# transpose()
t.T


# In[114]:


plt.figure(figsize=(15,7))
sns.heatmap(t.T, cmap="Blues", annot=True, fmt=".0f")


# In[118]:


g = df.groupby(["연도","지역명"])["평당분양가격"].mean().unstack().round()


# In[122]:


plt.figure(figsize=(15,7))
sns.heatmap(g.T, annot=True, fmt=".0f", cmap="Greens")


# ### 1.9 2013년부터 최근 데이터까지 시각화 하기

# In[123]:


# barplot 으로 연도별 평당분양가격 그리기 ci(신뢰구간)=95
sns.barplot(data=df, x="연도", y="평당분양가격")


# In[124]:


# pointplot 으로 연도별 평당분양가격 그리기
plt.figure(figsize=(12, 4))
sns.pointplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[125]:


# 서울만 barplot 으로 그리기
df_seoul = df[df["지역명"] == "서울"].copy()
print(df_seoul.shape)

sns.barplot(data=df_seoul, x="연도", y="평당분양가격", color="b")
sns.pointplot(data=df_seoul, x="연도", y="평당분양가격")


# In[126]:


# 연도별 평당분양가격 boxplot 그리기
sns.boxplot(data=df, x="연도", y="평당분양가격")


# In[127]:


sns.boxenplot(data=df, x="연도", y="평당분양가격")


# In[128]:


# 연도별 평당분양가격 violinplot 그리기
plt.figure(figsize=(10, 4))
sns.violinplot(data=df, x="연도", y="평당분양가격").set_title("연도별 전국 평균 평당 분양가격(단위:천원)")


# In[129]:


# 연도별 평당분양가격 swarmplot 그리기
plt.figure(figsize=(12, 5))
sns.swarmplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# * 지역별 평당분양가격 보기

# In[130]:


# barplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(12, 4))
sns.barplot(data=df, x="지역명", y="평당분양가격")


# In[131]:


# 위와 같은 그래프를 미리 연산을 하고 결과값으로 그리는 방법
# groupby 로 구하기
mean_price = df.groupby(["지역명"])["평당분양가격"].mean().to_frame().sort_values(by="평당분양가격", ascending=False)


# In[132]:


# groupby 로 했던 방법을 똑같이
# pivot_table 로 구하기
mean_price = df.pivot_table(index="지역명", values="평당분양가격").sort_values(by="평당분양가격", ascending=False)


# In[133]:


plt.figure(figsize=(12, 4))
sns.barplot(data=mean_price, x=mean_price.index, y="평당분양가격", palette="Blues_r")


# In[134]:


# boxplot 으로 지역별 평당분양가격을 그려봅니다.

plt.figure(figsize=(12, 4))
sns.boxplot(data=df, x="지역명", y="평당분양가격")


# In[135]:



plt.figure(figsize=(12, 4))
sns.boxenplot(data=df, x="지역명", y="평당분양가격")


# In[136]:


# violinplot 으로 지역별 평당분양가격을 그려봅니다.
plt.figure(figsize=(24, 4))
sns.violinplot(data=df, x="지역명", y="평당분양가격")


# In[137]:


# swarmplot 으로 지역별 평당분양가격을 그려봅니다.

plt.figure(figsize=(24, 4))
sns.swarmplot(data=df, x="지역명", y="평당분양가격", hue="연도")


# In[ ]:




