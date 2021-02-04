#!/usr/bin/env python
# coding: utf-8

# ## 소상공인시장진흥공단 상가업소정보로 스타벅스, 이디야 위치 분석하기
# 
# * 이디야는 스타벅스 근처에 입점한다는 설이 있습니다. 과연 이디야와 스타벅스의 매장입지는 얼마나 차이가 날까요? 관련 기사를 읽고 구별로 이디야와 스타벅스의 매장을 기사와 유사하게 분석하고 시각화 해보면서 Python, Pandas, Numpy, Seaborn, Matplotlib, folium 을 통해 다양한 방법으로 표현해 봅니다..
# 
# ### 다루는 내용
# * 공공데이터를 활용해 텍스트 데이터 정제하고 원하는 정보 찾아내기
# * 문자열에서 원하는 텍스트 추출하기
# * 문자열을 활용한 다양한 분석 방법과 위치 정보 사용하기
# * folium을 통한 위경도 데이터 시각화 이해하기
# 
# ### 실습
# * 텍스트 데이터 정제하기 -  대소문자로 섞여있는 상호명을 소문자로 변경하고 상호명 추출하기
# * 텍스트 데이터에서 원하는 정보 추출하기 - 브랜드명 컬럼을 만들고 구별 매장 수 분석하기
# * folium을 통해 지도에 분석한 내용을 표현하기 - CircleMarker와 choropleth 그리기
# 
# 
# ### 데이터셋
# * https://www.data.go.kr/dataset/15012005/fileData.do
# * 구별로 매장수를 표현하기 위해 GeoJSON 파일 로드
#     * 파일출처 : [southkorea/seoul-maps: Seoul administrative divisions in ESRI Shapefile, GeoJSON and TopoJSON formats.](https://github.com/southkorea/seoul-maps)
#     * 이 링크에서도 다운로드가 가능합니다. https://drive.google.com/open?id=13j8-_XLdPe0pptsqu8-uyE-0Ym6V2jw5
# 
# ### 관련기사
# * [[비즈&빅데이터]스타벅스 '쏠림' vs 이디야 '분산'](http://news.bizwatch.co.kr/article/consumer/2018/01/19/0015)

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


# 한글폰트 설정
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# 폰트가 선명하게 보이도록 retina 설정
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")


# In[38]:


# 한글폰트와 마이너스 폰트 설정 확인

pd.Series([-1, 0, 1, 3, 5]).plot(title="한글폰트")


# ## 데이터 불러오기

# In[39]:


df = pd.read_csv("data/상가업소정보_201912_01.csv", sep='|')
df.head()


# ## 5 데이터 불러오기

# In[40]:


df.shape


# In[41]:


df.info()


# In[42]:


# 결측치 보기
df.isnull().sum()


# In[43]:


# 결측치 비율보기 : mean()사용
df.isnull().mean().plot.barh(figsize=(7,9))


# In[44]:


# 사용하지 않는 컬럼 제거하기
# drop을 하는 방법도 있지만 사용할 컬럼만 따로 모아서 보는 방법도 있습니다.
# 여기에서는 사용할 컬럼만 따로 모아서 사용합니다.
columns = ['상호명', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명', 
           '시도명', '시군구명', '행정동명', '법정동명', '도로명주소', 
           '경도', '위도']

print(df.shape)
df = df[columns].copy()
df.shape


# In[45]:


df.info()


# ## 색인으로 서브셋 가져오기

# In[49]:


# 시도명이 서울로 시작하는 데이터만 봅니다.
# 또, df_seoul 이라는 변수에 결과를 저장합니다.
# 새로운 변수에 데이터프레임을 할당할 때 copy()를 사용하는 것을 권장합니다.
df_seoul = df[df["시도명"] == "서울특별시"].copy()
print(df_seoul.shape)
df_seoul.head()


# In[50]:


# unique를 사용하면 중복을 제거한 시군구명을 가져옵니다.
# 그리고 shape로 갯수를 출력해 봅니다.
df_seoul["시군구명"].unique()


# In[51]:


df_seoul["시군구명"].nunique()


# ## 7 파일로 저장하기
# * 전처리한 파일을 저장해 두면 재사용을 할 수 있습니다.
# * 재사용을 위해 파일로 저장합니다

# In[54]:


df_seoul.to_csv("seoul_open_store.csv", index = False)


# In[55]:


pd.read_csv("seoul_open_store.csv").head()


# ## 8 배스킨라빈스, 던킨도너츠 위치분석

# ### 특정 상호만 가져오기
# * 여기에서는 배스킨라빈스와 던킨도너츠 상호를 가져와서 실습합니다.
# * 위에서 pandas의 str.conatains를 활용해 봅니다.
# * https://pandas.pydata.org/docs/user_guide/text.html#testing-for-strings-that-match-or-contain-a-pattern
# 
# * 상호명에서 브랜드명을 추출합니다.
# * 대소문자가 섞여 있을 수도 있기 때문에 대소문자를 변환해 줍니다.
# * 오타를 방지하기 위해 배스킨라빈스의 영문명은 baskinrobbins, 던킨도너츠는 dunkindonuts 입니다.

# In[56]:


# 문자열의 소문자로 변경하는 메소드를 사용합니다.
# "상호명_소문자" 컬럼을 만듭니다.

df_seoul["상호명_소문자"] = df_seoul["상호명"].str.lower()


# In[57]:


df_seoul["상호명_소문자"].str.extract("(베|배)스킨라빈스|baskinrobbins")[0].value_counts()


# In[58]:


# baskinrobbins 를 "상호명_소문자" 컬럼으로 가져옵니다.
# 띄어쓰기 등의 다를 수 있기 때문에 앞글자 baskin 만 따서 가져오도록 합니다.
# '상호명_소문자'컬럼으로 '배스킨라빈스|baskin' 를 가져와 갯수를 세어봅니다.
# loc[행]
# loc[행, 열]

df_seoul.loc[df_seoul["상호명_소문자"].str.contains("배스킨라빈스|베스킨라빈스|baskinrobbins"), 
             "상호명_소문자"].shape


# In[59]:


# 상호명에서 던킨도너츠만 가져옵니다.
# 상호명은 소문자로 변경해 준 컬럼을 사용합니다.
# 던킨|dunkin 의 "상호명_소문자"로 갯수를 세어봅니다.

df_seoul.loc[df_seoul["상호명_소문자"].str.contains("던킨|dunkin"), "상호명_소문자"].shape


# In[60]:


# '상호명_소문자'컬럼으로  '배스킨|베스킨|baskin|던킨|dunkin'를 가져와 df_31 변수에 담습니다.

df_31 = df_seoul[df_seoul["상호명_소문자"].str.contains(
    '배스킨라빈스|베스킨라빈스|baskinrobbins|던킨|dunkin')].copy()
df_31.shape


# In[61]:


# ~은 not을 의미합니다. 베스킨라빈스가 아닌 데이터를 찾을 때 사용하면 좋습니다.
# 아래 코드처럼 결측치를 던킨도너츠로 채워줘도 괜찮습니다.
df_31.loc[df_31["상호명_소문자"].str.contains("배스킨라빈스|베스킨라빈스|baskinrobbins"), 
          "브랜드명"] = "배스킨라빈스"
df_31[["상호명", "브랜드명"]].head()


# In[62]:


# 'df_31에 담긴 상호명','브랜드명'으로 미리보기를 합니다.
# df_31.loc[~df_31["상호명_소문자"].str.contains("배스킨라빈스|베스킨라빈스|baskinrobbins"), 
#           "브랜드명"]

df_31["브랜드명"] = df_31["브랜드명"].fillna("던킨도너츠")
df_31["브랜드명"]


# In[63]:


# 데이터가 제대로 모아졌는지 확인합니다.
# "상권업종대분류명"을  value_counts 를 통해 빈도수를 계산합니다.

df_31["상권업종대분류명"].value_counts()


# In[64]:


# "상권업종대분류명"컬럼에서 isin 기능을 사용해서 "소매", "생활서비스" 인 데이터만 가져옵니다.

df_31[df_31["상권업종대분류명"].isin(["소매", "생활서비스"])]


# In[65]:


# "상권업종대분류명"에서 "소매", "생활서비스"는 제외합니다.
df_31 = df_31[~df_31["상권업종대분류명"].isin(["소매", "생활서비스"])].copy()
df_31.shape


# ### 범주형 값으로 countplot 그리기

# In[66]:


# value_counts 로 "브랜드명"의 빈도수를 구합니다.
brand_count = df_31["브랜드명"].value_counts()
brand_count


# In[67]:


# normalize=True 로 빈도수의 비율을 구합니다.
df_31["브랜드명"].value_counts(normalize=True).plot.barh()


# In[68]:


brand_count.index


# In[69]:


# countplot 을 그립니다.
g = sns.countplot(data=df_31, x="브랜드명")

for i, val in enumerate(brand_count.index):
    g.text(x=i, y=brand_count[i], s=brand_count[i])


# In[70]:


# 시군구명으로 빈도수를 세고 브랜드명으로 색상을 다르게 표현하는 countplot 을 그립니다.
plt.figure(figsize=(15, 4))
g = sns.countplot(data=df_31, x="시군구명", hue="브랜드명")


# In[71]:


# 위 그래프에도 숫자를 표시하고 싶다면 그룹화된 연산이 필요합니다.
# value_counts는 Series에만 사용이 가능하기 때문에 groupby 나 pivot_table로 구해볼 수 있습니다.
table_city_brand = df_31.pivot_table(index="시군구명", columns="브랜드명", values="상호명", aggfunc="count")
table_city_brand.head()


# In[72]:


# 그래프에 숫자를 표시하기 위해서는 하나씩 출력을 해봅니다.
# 데이터프레임을 순회할 수있는 iterrows() 을 사용해 보겠습니다.
# 아래에 출력되는 숫자를 그래프에 표시할 예정입니다.
for i, val in table_city_brand.iterrows():
    dunkin = val["던킨도너츠"]
    baskin = val["배스킨라빈스"]
    print(dunkin, baskin)


# In[73]:


plt.figure(figsize=(15, 4))
# 위에서 만든 피봇테이블과 "시군구명"의 순서과 같게 표시되게 하기 위해 order 값을 지정합니다.
g = sns.countplot(data=df_31, x="시군구명", hue="브랜드명", order=table_city_brand.index)

# 여기에서 i 값은 시군구명이라 숫자로 표현해줄 필요가 있습니다.
# 그래서 순서대로 0번부터 찍어줄 수 있도록 index_no 를 만들어 주고 for문을 순회할 때마다 하나씩 증가시킵니다.
index_no = 0
for i, val in table_city_brand.iterrows():
    dunkin = val["던킨도너츠"]
    baskin = val["배스킨라빈스"]
    g.text(x=index_no, y=dunkin, s=dunkin)
    g.text(x=index_no, y=baskin, s=baskin)
    index_no = index_no + 1


# ### scatterplot 그리기
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#scatter-plot

# In[74]:


# Pandas 의 plot 으로 scatterplot 을 그립니다.
df_31[["위도", "경도"]].plot.scatter(x="경도", y="위도")


# In[75]:


# seaborn의 scatterplot 으로 hue에 브랜드명을 지정해서 시각화 합니다.

sns.scatterplot(data=df_31, x="경도", y="위도", hue="브랜드명")


# In[76]:


# 위에서 그렸던 그래프를 jointplot 으로 kind="hex" 을 사용해 그려봅니다.

sns.jointplot(data=df_31, x="경도", y="위도", kind="hex")


# ## Folium 으로 지도 활용하기

# In[78]:


# 지도 시각화를 위한 라이브러리
import folium


# In[79]:


# 지도의 중심을 지정하기 위해 위도와 경도의 평균을 구합니다. 
lat = df_31["위도"].mean()
long = df_31["경도"].mean()
lat, long


# In[80]:


# 샘플을 하나 추출해서 지도에 표시해 봅니다.

m = folium.Map([lat, long])
# 127.039032	37.495593
folium.Marker(
    [37.495593, 127.039032], 
    popup='<i>던킨도너츠</i>', 
    tooltip="던킨도너츠").add_to(m)
m.save('index.html')
m


# In[81]:


# folium 사용법을 보고 일부 데이터를 출력해 봅니다.

df_31.sample(random_state=31)


# In[82]:


# html 파일로 저장하기
m.save('index.html')


# ### 서울의 배스킨라빈스와 던킨도너츠 매장 분포
# * 배스킨라빈스와 던킨도너츠 매장을 지도에 표현합니다.

# In[83]:


# 데이터프레임의 인덱스만 출력합니다.

df_31.index


# ### 기본 마커로 표현하기

# In[84]:


# icon=folium.Icon(color=icon_color) 로 아이콘 컬러를 변경합니다.

m = folium.Map([lat, long], zoom_start=12)

for i in df_31.index:
    
    sub_lat = df_31.loc[i, "위도"]
    sub_long = df_31.loc[i, "경도"]
    title = df_31.loc[i, "상호명"] + " - " + df_31.loc[i, "도로명주소"]
    
    icon_color = "blue"
    if df_31.loc[i, "브랜드명"] == "던킨도너츠":
        icon_color = "red"
    
    folium.Marker(
        [sub_lat, sub_long], 
        icon=folium.Icon(color=icon_color),
        popup=f'<i>{title}</i>', 
        tooltip=title).add_to(m)

m.save('index.html')
m


# ### MarkerCluster 로 표현하기
# * https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/MarkerCluster.ipynb

# In[85]:


# icon=folium.Icon(color=icon_color) 로 아이콘 컬러를 변경합니다.
from folium.plugins import MarkerCluster

m = folium.Map([lat, long], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for i in df_31.index:
    
    sub_lat = df_31.loc[i, "위도"]
    sub_long = df_31.loc[i, "경도"]
    title = df_31.loc[i, "상호명"] + " - " + df_31.loc[i, "도로명주소"]
    
    icon_color = "blue"
    if df_31.loc[i, "브랜드명"] == "던킨도너츠":
        icon_color = "red"
    
    folium.Marker(
        [sub_lat, sub_long], 
        icon=folium.Icon(color=icon_color),
        popup=f'<i>{title}</i>', 
        tooltip=title).add_to(marker_cluster)

m.save('index.html')
m


# ## 파리바게뜨와 뚜레주르 분석하기

# In[86]:


df_seoul["상호명"].str.extract("뚜레(주|쥬)르")[0].value_counts()


# In[88]:


# str.contains 를 사용해서 뚜레(주|쥬)르|파리(바게|크라상) 으로 상호명을 찾습니다.
# df_bread 라는 데이터프레임에 담습니다.
df_bread = df_seoul[df_seoul["상호명"].str.contains("뚜레(주|쥬)르|파리(바게|크라상)")].copy()
df_bread.shape


# In[89]:


# 잘못 가져온 데이터가 있는지 확인합니다.

df_bread["상권업종대분류명"].value_counts()


# In[90]:


# 제과점과 상관 없을 것 같은 상점을 추출합니다.

df_bread[df_bread["상권업종대분류명"] == "학문/교육"]


# In[91]:


# "상권업종대분류명"이 "학문/교육"이 아닌 것만 가져옵니다.

print(df_bread.shape)
df_bread = df_bread[df_bread["상권업종대분류명"] != "학문/교육"].copy()
print(df_bread.shape)


# In[92]:


# 상호명의 unique 값을 봅니다.

df_bread["상호명"].unique()


# In[93]:


df_bread[df_bread["상호명"].str.contains('파스쿠찌|잠바주스')]


# In[94]:


# 상호명이 '파스쿠찌|잠바주스'가 아닌 것만 가져오세요.
print(df_bread.shape)
df_bread = df_bread[~df_bread["상호명"].str.contains('파스쿠찌|잠바주스')].copy()
print(df_bread.shape)


# In[95]:


# 브랜드명 컬럼을 만듭니다. "파리바게뜨" 에 해당되는 데이터에 대한 값을 채워줍니다.

df_bread.loc[df_bread["상호명"].str.contains("파리바게"), "브랜드명"] = "파리바게뜨"
df_bread.loc[df_bread["상호명"].str.contains("파리크라상"), "브랜드명"] = "파리바게뜨"
df_bread.loc[df_bread["상호명"].str.contains("뚜레"), "브랜드명"] = "뚜레쥬르"
df_bread[["상호명", "브랜드명"]].head()


# In[96]:


df_bread["브랜드명"].isnull().value_counts()


# In[97]:


df_bread[df_bread["브랜드명"].isnull() == True]


# In[98]:


# 브랜드명 컬럼의 결측치는 "뚜레쥬르" 이기 때문에 fillna 를 사용해서 값을 채웁니다.

df_bread["브랜드명"] = df_bread["브랜드명"].fillna("뚜레쥬르")
df_bread[["상호명", "브랜드명"]].head()


# In[99]:


# 브랜드명의 빈도수를 봅니다.

df_bread["브랜드명"].value_counts()


# In[100]:


df_bread["브랜드명"].value_counts(normalize=True)


# In[101]:


# countplot 으로 브랜드명을 그려봅니다.

sns.countplot(data=df_bread, x="브랜드명")


# In[102]:


# 시군구별로 브랜드명의 빈도수 차이를 비교합니다.
plt.figure(figsize=(15, 4))
sns.countplot(data=df_bread, x="시군구명", hue="브랜드명")


# In[103]:


# scatterplot 으로 위경도를 표현해 봅니다.
sns.scatterplot(data=df_bread, x="경도", y="위도", hue="브랜드명")


# In[104]:


# jointplot 으로 위경도를 표현해 봅니다.
sns.jointplot(data=df_bread, x="경도", y="위도", kind="hex")


# ## 지도에 표현하기
# ### Marker 로 위치를 찍어보기

# In[105]:


df_bread.index


# In[106]:


df_bread.loc[2935, "위도"]


# In[107]:


# icon=folium.Icon(color=icon_color) 로 아이콘 컬러를 변경합니다.
m = folium.Map([lat, long], zoom_start=12, tiles="stamen toner")

for i in df_bread.index:
    sub_lat = df_bread.loc[i, "위도"]
    sub_long = df_bread.loc[i, "경도"]
    
    title = df_bread.loc[i, "상호명"] + " - " + df_bread.loc[i, "도로명주소"]
    
    icon_color = "blue"
    if df_bread.loc[i, "브랜드명"] == "뚜레쥬르":
        icon_color = "green"

    folium.CircleMarker(
        [sub_lat,sub_long ], 
        radius=3,
        color=icon_color,
        popup=f'<i>{title}</i>', 
        tooltip=title).add_to(m)

m.save('paris-tour.html')
m


# ### MarkerCluster 로 표현하기

# In[108]:


m = folium.Map([lat, long], zoom_start=12, tiles="stamen toner")

marker_cluster = MarkerCluster().add_to(m)

for i in df_bread.index:
    sub_lat = df_bread.loc[i, "위도"]
    sub_long = df_bread.loc[i, "경도"]
    
    title = df_bread.loc[i, "상호명"] + " - " + df_bread.loc[i, "도로명주소"]
    
    icon_color = "blue"
    if df_bread.loc[i, "브랜드명"] == "뚜레쥬르":
        icon_color = "green"

    folium.CircleMarker(
        [sub_lat,sub_long ], 
        radius=3,
        color=icon_color,
        popup=f'<i>{title}</i>', 
        tooltip=title).add_to(marker_cluster)

m.save('paris-tour.html')
m


# ### Heatmap 으로 그리기
# https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Heatmap.ipynb

# In[109]:


data = (
    np.random.normal(size=(100, 3)) *
    np.array([[1, 1, 1]]) +
    np.array([[48, 5, 1]])
).tolist()
data[:5]


# In[110]:


# heatmap 예제와 같은 형태로 데이터 2차원 배열 만들기
heat = df_bread[["위도", "경도", "브랜드명"]].copy()
heat["브랜드명"] = heat["브랜드명"].str.strip()
heat["브랜드명"] = heat["브랜드명"].replace("뚜레쥬르", 1).replace("파리바게뜨", 1)
heat = heat.values
# heat


# In[111]:


heat[:5]


# In[112]:


# HeatMap 그리기
from folium.plugins import HeatMap

m = folium.Map([lat, long], tiles='stamentoner', zoom_start=12)


for i in df_bread.index:
    sub_lat = df_bread.loc[i, "위도"]
    sub_long = df_bread.loc[i, "경도"]
    
    title = df_bread.loc[i, "상호명"] + " - " + df_bread.loc[i, "도로명주소"]
    
    icon_color = "blue"
    if df_bread.loc[i, "브랜드명"] == "뚜레쥬르":
        icon_color = "green"

    folium.CircleMarker(
        [sub_lat,sub_long ], 
        radius=3,
        color=icon_color,
        popup=f'<i>{title}</i>', 
        tooltip=title).add_to(m)

    
HeatMap(heat).add_to(m)

m.save('Heatmap.html')

m


# In[113]:


df_academy = df[(df["상권업종대분류명"] == '학문/교육') & (df["시군구명"] == "종로구")]


# In[114]:


df_academy.groupby(['시군구명'])['상권업종소분류명'].value_counts()


# In[115]:


df_academy.groupby(['시군구명','상권업종소분류명'])['상호명'].count().sort_values(ascending=False)


# In[ ]:




