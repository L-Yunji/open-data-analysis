#!/usr/bin/env python
# coding: utf-8

# # 대형마트와 전통시장 입점 분석

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# 한글, -(마이너스) 깨짐 없이 보기
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)


# In[4]:


# 폰트가 선명하게 보이도록 설정
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")


# In[5]:


pd.Series([-1, 0, 1, 3, 5]).plot(title="한글폰트")


# In[6]:


df = pd.read_csv("data/서울시 전통시장 현황.csv", encoding='CP949')
df.shape


# In[7]:


df.head(1)


# In[9]:


df.isnull().sum()


# In[11]:


df.info()


# In[12]:


columns = ['자치구명', '전통시장명', '주소명', '경도', '위도']

df = df[columns].copy()
df.shape


# In[13]:


df.info()


# In[14]:


df.columns = ['구', '시장명', '주소명', '경도', '위도']
df.head()


# In[15]:


df_market = df["시장명"].str.contains("시장")
df_market.value_counts()


# In[28]:


df_mk = df.loc[df["시장명"].str.contains("시장")]
df_mk.head(10)


# In[29]:


plt.figure(figsize=(15, 4))
sns.countplot(data=df_mk, x="구")


# In[30]:


df_mk[["위도", "경도"]].plot.scatter(x="경도", y="위도")


# In[31]:


sns.scatterplot(data=df_mk, x="경도", y="위도", hue="구")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[32]:


df_mk["구"].value_counts().head(10)


# In[33]:


df_mk


# In[42]:


dff = pd.read_csv('data/전통시장현황_20200226..csv', encoding='CP949')
dff.shape


# In[45]:


dff.info()


# In[44]:


dff.isnull().sum().plot.barh(figsize=(10,25))


# In[46]:


dff.columns


# In[47]:


columns = ['시장명','시군구','시도','보유갯수 - 16시장전용 고객주차장',
           '시장/상점가의 주력 상품 여부(1=있음, 2=없음)','보유현황 - 10쇼핑카트(1=있음, 2=없음)',
           '시장/상점가의 주력 상품의 상품명']


# In[48]:


dff = dff[columns].copy()
dff.shape


# In[50]:


dff.info()


# In[51]:


df_seoul = dff[dff["시도"] == "서울"].copy()
df_seoul.shape


# In[98]:


df_seoul = df_seoul[df_seoul["시장명"].str.contains("시장")]
df_seoul


# In[70]:


df_mk


# In[102]:


if df_mk.loc["시장명"] == df_seoul.loc["시장명"]:
    df = pd.concat([df_mk, df_seoul], axis=1)
    
df


# In[77]:


df_final = df_mk["시장명"] == df_seoul["시장명"]
df_final


# In[107]:


df = pd.merge(df_mk, df_seoul, left_on='시장명', right_on='시장명').copy()
    
df.head()


# In[100]:


df["시장명"]=="강남시장"


# In[93]:


for i in range():
    if df_mk.loc[i, "시장명"] == df_seoul.loc[i, "시장명"]
        df = pd.concat([df_mk, df_seoul])
    
df


# In[91]:


df_mk.loc[i, "시장명"]


# In[97]:


for i in df_seoul.index:
    print(df_seoul.loc[i, "시장명"])


# In[22]:


import folium


# In[34]:


long = df_mk["경도"].mean()
lat = df_mk["위도"].mean()


# In[37]:


m = folium.Map([lat,long], zoom_start=12)

for i in df_mk.index:
    sub_lat = df_mk.loc[i, "위도"]
    sub_long = df_mk.loc[i, "경도"]
    title = df_mk.loc[i, "시장명"] +"("+ df_mk.loc[i, "주소명"]+")"
    
    icon_color = "red"
    
    folium.Marker([sub_lat, sub_long],
                  icon=folium.Icon(color=icon_color),
                  tooltip=title,
                  popup=f'<i>{title}</i>').add_to(m)

# for문 끝나고 지도 출력
m


# In[25]:


m = folium.Map(location=[lat, long], zoom_start=12, tiles="Stamen Toner")
# 다른 스타일 사용 : tiles="Stamen Toner"

for i in df_m.index[:100]:
    tooltip = df_m.loc[i, "시장명"] +"("+ df_m.loc[i, "주소명"]+")"
    lat = df_m.loc[i, "위도"]
    long = df_m.loc[i, "경도"]

    #folium.Marker([lat, long], tooltip=tooltip).add_to(m)
    folium.CircleMarker([lat, long], tooltip=tooltip, radius=3).add_to(m)
    
m


# In[26]:


# 겹친 아이콘 처리
from folium.plugins import MarkerCluster

# icon=folium.Icon(color=icon_color) 로 아이콘 컬러를 변경합니다.

m = folium.Map([lat,long], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for i in df_m.index:
    sub_lat = df_m.loc[i, "위도"]
    sub_long = df_m.loc[i, "경도"]
    title = df_m.loc[i, "시장명"]+" ("+df_m.loc[i, "주소명"]+")"
    icon_color = "orange"
    
    folium.Marker([sub_lat, sub_long],
                  icon=folium.Icon(color=icon_color),
                  popup=f'<i>{title}</i>', 
                  tooltip=title).add_to(marker_cluster)

m


# In[27]:


m = folium.Map([lat, long], zoom_start=11, tiles='stamen toner')

folium.Choropleth(
    geo_data=geo_json,
    name="choropleth",
    data=df_m,
    columns=["구", "시장수"],
    key_on="feature.properties.name",
    fill_color="BuGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="시장수 비교",
).add_to(m)
 
for i in df_m.index:
    sub_long =  df_m.loc[i, "경도"]
    sub_lat =  df_m.loc[i, "위도"]
    
    markets = ["대형마트", "시장"] # 원 두 개 그리기 : 리스트
    for cafe in cafes:
        cafe_count = df_vs.loc[i, cafe]
        
        radius = np.sqrt(cafe_count)*3
        
        gu = df_vs.loc[i, "구"]
        tooltip = f"{gu} ({cafe} : {cafe_count})"
        
        color ="green"
        if cafe=="이디야":
            sub_long = sub_long+0.01 # 비껴 그리도록
            color="blue"
        
        folium.CircleMarker([sub_lat, sub_long], 
                            radius = radius,
                            fill=True,
                            color=color, tooltip = tooltip
                           ).add_to(m)
    
m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


from pyproj import Proj, transform


# In[ ]:


# Projection 정의
# UTM-K
proj_UTMK = Proj(init='epsg:5178') # UTM-K(Bassel) 도로명주소 지도 사용 중

# WGS1984
proj_WGS84 = Proj(init='epsg:4326') # Wgs84 경도/위도, GPS사용 전지구 좌표

# UTM-K -> WGS84 샘플
x1, y1 = 961114.519726,1727112.269174
x2, y2 = transform(proj_UTMK,proj_WGS84,x1,y1)
print(x2,y2)

# WGS84 -> UTM-K 샘플
x1, y1 = 127.07098392510115, 35.53895289091983
x2, y2 = transform(proj_WGS84, proj_UTMK, x1, y1)
print(x2,y2)

# x, y 컬럼을 이용하여 UTM-K좌표를 WGS84로 변환한 Series데이터 반환
def transform_utmk_to_w84(df):
    return pd.Series(transform(proj_UTMK, proj_WGS84, df['x'], df['y']), index=['x', 'y'])

df_xy = pd.DataFrame([
                                        ['A', 961114.519726,1727112.269174],
                                        ['B', 940934.895125,1685175.196487],
                                        ['C', 1087922.228298,1761958.688262]
                                    ], columns=['id', 'x', 'y'])

df_xy[['x_w84', 'y_w84']] = df_xy.apply(transform_utmk_to_w84, axis=1)


# In[64]:


dff = pd.read_csv("data/상가업소정보_201912_01.csv", sep='|')
dff.shape

