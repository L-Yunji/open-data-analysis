# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")
# - 값 설정
plt.rc("axes", unicode_minus=False)

# +
# 레티나 디스플레이로 폰트가 선명하게 표ㅣ되도록 합니다.

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
# -

# ### 3.1 한글 폰트 설정 확인

pd.Series([-4, 1, 0, 3, -2, 4, 5]).plot(title="한글폰트 설정")

# ### 파일 로드하기

# %ls data

# Parser Error sep ="|"로 분리
df = pd.read_csv("data/상가업소정보_201912_01.csv", sep="|")
df.shape

#모든 컬럼이 표시되도록 max_columns의 수를 지정합니다.
pd.options.display.max_columns = 39

df.head()

df.tail()

# ### 인덱스 정보 보기

df.index

df.columns

# ### 5.4 info

# 데이터 타입 써져있고~ 데이터 결측치 뺀 갯수 값도 적혀있다.
df.info()

# ### 5.5 데이터 타입 보기

df.dtypes

# ### 6 결측치 확인하기

# 결측치를 구해서 n이라는 변수에 담고 재사용 합니다. true로 표시 된 것이 결측치!
df.isnull()

n = df.isnull().sum()

n.plot.bar()

# 값을 정렬해서 결측치가 많은 값이 위에 그려지도록 barh로 그립니다.
# sort_values() 값을 내림차순으로 정리
n.sort_values().plot.barh(figsize=(7,8))

# ### 6.2 missingno 로 결측치 시각화 하기

import missingno as msno
msno.matrix(df)

# heatmap으로 표현합니다. 상관관계가 1일수록 양의 상관관계
msno.heatmap(df)

msno.dendrogram(df)

# ### 7 사용하지 않는 컬럼 제거하기

# #### 7.1 결측치가 너무 많은 컬럼 제거하기

#sort_values 를 통해 결측치가 많은 데이터를 위에서 9개 가져와서 not_use 변수에 담습니다.
# not_use 변수에 담긴 인덱스값만 추출해서 not_use_col 이라는 변수에 담습니다.
not_use = n.sort_values(ascending=False).head(9)
not_use_col = not_use.index
not_use_col

print(df.shape)
df = df.drop(not_use_col, axis=1)
print(df.shape)

df.info()

# ### 7.2 사용하지 않는 컬럼 제거하기

cols = df.columns
cols

cols_code = cols[cols.str.contains("코드|번호")]
cols_code

print(df.shape)
df = df.drop(cols_code, axis = 1)
print(df.shape)

df.info()

# ### 8 행, 열을 기준으로 값을 가져오기
# * df[열이름]
#     * 결과가 Pandas 의 Series 형태로 반환
# * df[[열목록]] : 2개 이상의 열을 가져올 때는 리스트 형태로 묶어서 지정해주어야 합니다.
#     * 결과가 Pandas 의 DataFrame 형태로 반환
#     * 1개의 열을 2차원 리스트로 지정할 때에도 DataFrame 형태로 반환이 됩니다.

# "상호명" 컬럼만 가져옵니다.
df["상호명"].head()

# :상호명", "도로명주소" 2개의 칼럼을 가져옵니다. 2개이상 들고오면 []에 넣어야 한다.
# [[]]면 dataframe 형태로 가져와 진다.
df[["상호명", "도로명주소"]]

# ### 행(row) 기준
# * df.loc[행]
# * df.loc[행, 열]
#
#
# * df.loc[행이름 혹은 번호]
#     * 결과가 Pandas 의 Series 형태로 반환
# * df.loc[[행목록]] : 2개 이상의 행을 가져올 때는 열(column)을 가져올 때와 마찬가지로 리스트 형태로 묶어서 지정해주어야 합니다.
#     * 결과가 Pandas 의 DataFrame 형태로 반환
#     * 1개의 행을 2차원 리스트로 지정할 때에도 DataFrame 형태로 반환이 됩니다.

df.loc[0]

df.loc[[0,1,2]]

#df.loc[0]["상호명"]
df.loc[0, "상호명"]

df.loc[[0,1,2], ["상호명","도로명주소"]]

# ### iloc 로 슬라이싱을 통해 가져오기
#
# * df.iloc[:] 전체 데이터를 가져옵니다.  
# * df.iloc[행, 열] 순으로 인덱스 번호를 지정합니다.
# * : 은 전체를 의미합니다.
# * 시작인덱스:끝나는인덱스+1을 써줍니다. 
#     * 예) 3:5 라면 3번째 인덱스 부터 4번째 인덱스까지 가져옵니다.
# * : 에서 앞이나 뒤 인덱스를 써주지 않으면 처음부터 혹은 끝까지를 의미합니다.
#     * 예) :5 => 처음부터 4번 인덱스까지 가져옵니다.
#     * 예) 5: => 5번 인덱스부터 끝까지 가져옵니다.
#     * 예) -5: => 뒤에서 5번째 부터 끝까지 가져옵니다.
#     * 예) :-5 => 처음부터 5번째 전까지 가져옵니다.

#위에서 5개의 행과, 왼쪽에서 5개의 열을 가져옵니다.
df.iloc[:5, :5]

# 끝에서 5개의 행과, 오른쪽에서 5개의 열을 가져옵니다.
df.iloc[-5:, -5:]

# ### describe 로 요약하기
#
#
# * describe 를 사용하면 데이터를 요약해 볼 수 있습니다.
# * 기본적으로 수치형 데이터를 요약해서 보여줍니다. 
# * include, exclude 옵션으로 데이터 타입에 따른 요약수치를 볼 수 있습니다.
# * 데이터의 갯수, 평균, 표준편차, 최솟값, 1사분위수(25%), 2사분위수(50%), 3사분위수(75%), 최댓값을 볼 수 있습니다.
# * [Descriptive statistics - Wikipedia](https://en.wikipedia.org/wiki/Descriptive_statistics)

# describe 로 기술통계값을 봅니다. 수치형 데이터의 기술통계값을 보여준다.
# DataFrame.count: Count number of non-NA/null observations.
# DataFrame.max: Maximum of the values in the object.
# DataFrame.min: Minimum of the values in the object.
# DataFrame.mean: Mean of the values.
# DataFrame.std: Standard deviation of the observations.
# DataFrame.select_dtypes: Subset of a DataFrame including/excluding
#     columns based on their dtype.
df.describe()

# 필요한 컬럼에 대한 요약만 합니다.
# 위도, 경도 만 따로 가져와 요약합니다.
df[["위도", "경도"]].describe()

# ### 개별 기술통계 값 구하기
#
#
#
# * [Computational tools — pandas 1.0.1 documentation](https://pandas.pydata.org/docs/user_guide/computation.html#method-summary)
#
# * count 결측치를 제외한 (NaN이 아닌) 값의 갯수를 계산
# * min, max 최솟값, 최댓값
# * argmin, argmax 최솟값, 최댓값이 위치한 (정수)인덱스를 반환
# * idxmin, idxmax 인덱스 중 최솟값, 최댓값을 반환
# * quantile 특정 사분위수에 해당하는 값을 반환 (0~1 사이)
#     * 0.25 : 1사분위 수
#     * 0.5 : 2사분위수 (quantile 의 기본 값)
#     * 0.75 : 3사분위수
# * sum 수치 데이터의 합계
# * mean 평균
# * median 중앙값(중간값:데이터를 한 줄로 세웠을 때 가운데 위치하는 값, 중앙값이 짝수일 때는 가운데 2개 값의 평균을 구함)
# * mad 평균값으로부터의 절대 편차(absolute deviation)의 평균
# * std, var 표준편차, 분산을 계산
# * cumsum 맨 첫 번째 성분부터 각 성분까지의 누적합을 계산 (0 번째 부터 계속 더해짐)
# * cumprod 맨 첫번째 성분부터 각 성분까지의 누적곱을 계산 (1 번째 부터 계속 곱해짐)

# 결측치를 제외한 (Nan이 아닌)값의 갯수를 계산 
df["위도"].count()

df["위도"].mean()

df["위도"].max()

df["위도"].min()

df["위도"].median()

# 분산
df["위도"].var()

# * 출처 : [표준 편차 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%ED%91%9C%EC%A4%80_%ED%8E%B8%EC%B0%A8)
#
# 표준 편차(標準 偏差, 영어: standard deviation)는 자료의 산포도를 나타내는 수치로, 분산의 양의 제곱근으로 정의된다. 표준편차가 작을수록 평균값에서 변량들의 거리가 가깝다.[1] 통계학과 확률에서 주로 확률의 분포, 확률변수 혹은 측정된 인구나 중복집합을 나타낸다. 일반적으로 모집단의 표준편차는 {\displaystyle \sigma }\sigma (시그마)로, 표본의 표준편차는 {\displaystyle S}S(에스)로 나타낸다.[출처 필요]
#
# 편차(deviation)는 관측값에서 평균 또는 중앙값을 뺀 것이다.
#
# 분산(variance)은 관측값에서 평균을 뺀 값을 제곱하고, 그것을 모두 더한 후 전체 개수로 나눠서 구한다. 즉, 차이값의 제곱의 평균이다. 관측값에서 평균을 뺀 값인 편차를 모두 더하면 0이 나오므로 제곱해서 더한다.
#
# 표준 편차(standard deviation)는 분산을 제곱근한 것이다. 제곱해서 값이 부풀려진 분산을 제곱근해서 다시 원래 크기로 만들어준다.

import numpy as np
np.sqrt(df["위도"].var())

df["위도"].std()

# ### 9.3 단변량 수치형 변수 시각화

# 위도의 displot을 그립니다.
sns.distplot(df["위도"])

# 경도의 distplot을 그립니다.
sns.distplot(df["경도"])

plt.axvline(df["위도"].mean(), linestyle=":", color="r")
plt.axvline(df["위도"].median(), linestyle="--")
sns.distplot(df["위도"])

# ### 상관계수
# * [상관 분석 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%83%81%EA%B4%80_%EB%B6%84%EC%84%9D)
# * r 값은 X 와 Y 가 완전히 동일하면 +1, 전혀 다르면 0, 반대방향으로 완전히 동일 하면 –1 을 가진다.
# * 결정계수(coefficient of determination) 는 r ** 2 로 계산하며 이것은 X 로부터 Y 를 예측할 수 있는 정도를 의미한다.
#     * r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
#     * r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
#     * r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
#     * r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
#     * r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
#     * r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
#     * r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
#     
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/800px-Correlation_examples2.svg.png" width="600">
#
# 이미지 출처 : [Correlation and dependence - Wikipedia](https://en.wikipedia.org/wiki/Correlation_and_dependence)

# * 피어슨 상관 계수 : r = X 와 Y가 함께 변하는 정도 / X 와 Y가 각각 변하는 정도 

# 각 변수의 상관계수를 구합니다.
corr = df.corr()
corr

mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, annot=True, cmap="Blues", mask=mask)

# ### 산점도로 이변량 수치형 변수 표현하기

# scatterplot 으로 경도와 위도를 표현하며, 
# 데이터의 갯수가 많으면 오래 걸리기 때문에 1000 개의 샘플을 추출해서 그립니다.
sns.scatterplot(data=df.sample(1000), x="경도", y="위도")

# 위 시각화에서 회귀선을 그립니다.
sns.regplot(data=df.sample(1000), x="경도", y="위도")

sns.lmplot(data=df.sample(1000), x="경도", y="위도", hue="시도명", col="시도명",truncate=False, fit_reg=True)

# ### 9.6 object타입의 데이터 요약하기

#include="object"로 문자열 데이터에 대한 요약을 합니다.
df.describe(include="object")

#상권업종대분류명의 요약값을 봅니다.
df["상권업종대분류명"].describe()

df["상권업종대분류명"].unique()

df["상권업종대분류명"].nunique()

# 최빈값 구하기
df["상권업종대분류명"].mode()

df["상권업종대분류명"].value_counts()

# ### 10 인덱싱과 필터로 서브셋 만들기

# 엑셀로는 큰 데이터 다루기가 힘들다.
# "상권업종대분류명" 이 "음식인 데이터만 가져오기
# df_food 라는 변수에 담아줍니다. 서브셋을 새로운 변수에 저장할 때 copy()를 사용하는 것을 권장합니다.
df_food = df[df["상권업종대분류명"] == "음식"].copy()
df_food.head()

#시군구명이 "강남구"이고 "상권업종대분류명"이 "음식"인 서브셋을 구한 후
#"상권업종중분류명"별로 빈도수를 구합니다.
df[(df["시군구명"] == "강남구") & (df["상권업종대분류명"] == "음식")]["상권업종중분류명"].value_counts()

# 위와 똑같이 구하지만 이번에는 loc를 사용합니다.
# loc[행, 열]
# 훨씬 빠르다
df.loc[(df["시군구명"] == "강남구") & (df["상권업종대분류명"] == "음식"),
       "상권업종중분류명"].value_counts()

# +
# df_seoul_food 에 "시도명"이 "서울특별시" 이고 "상권업종대분류명" 이 "음식" 에 대한 서브셋만 가져와서 담아줍니다.

df_seoul_food = df[(df["시도명"] == "서울특별시")& (df["상권업종대분류명"] == "음식")].copy()
df_seoul_food.shape
# -

# "시군구명", "상권업종중분류명" 으로 그룹화 해서 상점수를 세어봅니다.
# 결과를 food_gu 에 담아 재사용할 예정입니다.
food_gu = df_seoul_food.groupby(["시군구명","상권업종중분류명"])["상호명"].count()
food_gu.head()

# food_gu 에 담긴 데이터를 시각화 합니다.
# 상권업종중분류명 과 상점수 로 barplot을 그립니다.
food_gu.unstack().loc["강남구"].plot.bar()

food = food_gu.reset_index()
food = food.rename(columns={"상호명":"상호수"})
food.head()

plt.figure(figsize=(15,4))
sns.barplot(data=food, x="상권업종중분류명", y="상호수")

# catplot을 사용하여 서브프롯을 그립니다.
sns.catplot(data=food, x="상권업종중분류명", y="상호수", kind="bar", 
            col="시군구명", col_wrap=4)

# ## 구별 학원수 비교
# * 주거나 입지로 문화시설이나 학원, 교육 등을 고려하게 됩니다.
# * 사교육이 발달한 지역으로 대치동이나 목동을 꼽는데 이 지역에 학원이 많이 분포 되어 있는지 알아봅니다.

# ### 서브셋 만들고 집계하기

# 학원의 분류명을 알아보기 위해 "상권업종대분류명"의 unique 값을 추출합니다.
df["상권업종대분류명"].unique()

# +
# "시도명"이 "서울특별시"이고 "상권업종대분류명"이 "학문/교육" 인 데이터를 서브셋으로 가져옵니다.
# 재사용을 위해 서브셋을 df_academy 에 저장합니다.

df_academy = df[(df["시도명"] == "서울특별시") & (df["상권업종대분류명"] == "학문/교육")].copy()
df_academy.head()
# -

# df_academy 에서 "상호명"으로 빈도수를 구합니다.
df_academy["상호명"].value_counts().head(10)

# "시군구명"으로 빈도수를 구합니다.
df_academy["시군구명"].value_counts()

# "상권업종소분류명"으로 빈도수를 구함
academy_count = df_academy["상권업종소분류명"].value_counts().head(30)
academy_count

# "상권업종소분류명"으로 빈도수를 구하고
# 빈도수가 1000개 이상인 데이터만 따로 봅니다.
academy_count_1000 = academy_count[academy_count > 1000]
academy_count_1000

# "시군구명", "상권업종소분류명" 으로 그룹화를 하고 "상호명"으로 빈도수를 계산합니다.
academy_group = df_academy.groupby(["시군구명","상권업종소분류명"])["상호명"].count()
academy_group.loc["강남구"]

t = academy_group.reset_index()
t = t.rename(columns={"상호명":"상호수"})
t.head()

# ### seaborn으로 시각화 하기

# 위에서 구한 결과를 시군구명, 상호수로 barplot을 그립니다.
plt.figure(figsize=(15,3))
sns.barplot(data=t, x="시군구명", y="상호수", ci=None)

# ### 11.3 isin 을 사용해 서브셋 만들기
# * 상권업종소분류명을 빈도수로 계산했을 때 1000개 이상인 데이터만 가져와서 봅니다.

academy_count_1000.index

# isin 으로 빈도수로 계산했을 때 1000개 이상인 데이터만 가져와서 봅니다.
# 서브셋을 df_academy_selected 에 저장합니다.
print(df_academy.shape)
df_academy_selected = df_academy[df_academy["상권업종소분류명"].isin(academy_count_1000.index)].copy()
df_academy_selected.shape

df_academy_selected["상권업종소분류명"].value_counts()

# df_academy_selected 의 "시군구명"으로 빈도수를 셉니다.
df_academy_selected["시군구명"].value_counts()

df_academy_selected.loc[
    df_academy_selected["법정동명"] == "대치동",
    "상권업종소분류명"].value_counts()

df_academy_selected.loc[
    df_academy_selected["법정동명"] == "목동",
    "상권업종소분류명"].value_counts()

# df_academy_selected 로 위에서 했던 그룹화를 복습해 봅니다.
# "상권업종소분류명", "시군구명" 으로 그룹화를 하고 "상호명"으로 빈도수를 계산합니다.
# g 라는 변수에 담아 재사용 할 예정입니다. 
g = df_academy_selected.groupby(["상권업종소분류명","시군구명"])["상호명"].count()
g

# ### Pandas 의 plot 으로 시각화

# 상권업종소분류명이 index 로 되어 있습니다.
# loc를 통해 index 값을 가져올 수 있습니다.
# 그룹화된 결과 중 "학원-입시" 데이터만 가져옵니다. 
g.loc["학원-입시"].sort_values().plot.barh(figsize=(10,7))

# +
# 그룹화된 데이터를 시각화 하게 되면 멀티인덱스 값으로 표현이 되어 보기가 어렵습니다.
# 다음 셀부터 이 그래프를 개선해 봐요!

g.plot()
# -

# ### unstack() 이해하기
# * https://pandas.pydata.org/docs/user_guide/reshaping.html
# <img src="https://pandas.pydata.org/docs/_images/reshaping_stack.png">
# <img src="https://pandas.pydata.org/docs/_images/reshaping_unstack.png">

# 위에서 그룹화한 데이터를 unstack() 하고 iloc로 위에서 5개 왼쪽에서 5개만 서브셋을 봅니다.
g.unstack().iloc[:5,:5]

g.unstack().loc["학원-입시"].plot.barh(figsize=(8,9))

g.unstack().T.plot.bar(figsize=(15,5))

g.index

# 멀티인덱스보다 컬럼으로 접근이 편하기 때문에 reset_index 를 통해 인덱스값을 컬럼으로 만들어 줍니다.
# "상호명" 컬럼은 "상호수" 이기 때문에 컬럼명을 변경해 줍니다.
t = g.reset_index()
t = t.rename(columns={"상호명":"상호수"})
t

# ### 같은 그래프를 seaborn 으로 그리기
# * pandas는 index값을 x축에 그린다. 
# * seaborn으로 그릴 때는 melt
# <img src="https://pandas.pydata.org/pandas-docs/stable/_images/reshaping_melt.png">
#
# 이미지 출처 : https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html

# x축에 시군구명을 y축에 상호수를 막대그래프로 그립니다. 
# 상권업종소분류명 으로 색상을 다르게 표현합니다.
plt.figure(figsize=(15,4))
sns.barplot(data=t, x="시군구명", y="상호수", ci=None)

# x축에 상권업종소분류명을 y축에 상호수를 막대그래프로 그립니다. 
# 시군구명 으로 색상을 다르게 표현합니다.
plt.figure(figsize=(15,4))
sns.barplot(data=t, x="상권업종소분류명", y="상호수", ci=None)

# "상권업종소분류명"이 "학원-입시" 인 서브셋만 가져와서 시각화 합니다.
academy_sub = t[t["상권업종소분류명"] == "학원-입시"].copy()
print(academy_sub.shape)
plt.figure(figsize=(15,4))
sns.barplot(data=academy_sub, x="시군구명", y="상호수")

# catplot을 통해 서브플롯을 그립니다.
sns.catplot(data=t, x="상권업종소분류명", y="상호수", kind="bar", 
            col="시군구명", col_wrap=4, sharex=False)

# ### 경도와 위도를 scatterplot 으로 표현하기

# scatterplot 으로 경도와 위도를 표현하고 시군구명으로 색상을 다르게 표현합니다
plt.figure(figsize=(10,7))
sns.scatterplot(data=df_academy_selected, x="경도", y="위도", hue="시군구명")

plt.figure(figsize=(10,7))
sns.scatterplot(data=df_academy_selected, x="경도", y="위도", hue="상권업종소분류명")

# "상권업종소분류명"이 "학원-입시" 인 데이터만 그려봅니다.
plt.figure(figsize=(10,7))
sns.scatterplot(
    data=df_academy_selected[df_academy_selected["상권업종소분류명"] == "학원-입시"],
    x="경도", y="위도", hue="상권업종소분류명")

plt.figure(figsize=(10,7))
sns.scatterplot(
    data=df_academy_selected[df_academy_selected["상권업종소분류명"] == "어린이집"],
    x="경도", y="위도", hue="상권업종소분류명")

# 어린이집과 학원-입시를 비교해 봅니다.
plt.figure(figsize=(10,7))
sns.scatterplot(
    data=df_academy_selected[
        df_academy_selected["상권업종소분류명"].isin(["어린이집","학원-입시"])],
    x="경도", y="위도", hue="상권업종소분류명")

#
# ## Folium 으로 지도 활용하기
#
#
# ### Folium 사용예제
# * 예제목록 : http://nbviewer.jupyter.org/github/python-visualization/folium/tree/master/examples/
# * Quickstart : https://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Quickstart.ipynb

import folium

long = df_academy_selected["경도"].mean()
lat = df_academy_selected["위도"].mean()

df_m = data=df_academy_selected[
        df_academy_selected["상권업종소분류명"].isin(["어린이집","학원-입시"])]
df_m = df_m.sample(1000)
df_m.shape

df_m.iloc[0]

# +
m = folium.Map(location=[lat, long], zoom_start=12)

folium.Marker(
    [37.4978, 127.061], tooltip="박영배수학교습소"
).add_to(m)

m
# -

m.save('index.html')

df_m.loc[95765,"상호명"]

df_m.index

# +
m = folium.Map(location=[lat, long], zoom_start=12,
    tiles="Stamen Toner")

for i in df_m.index[:10]:
    tooltip = df_m.loc[i,"상호명"] + "-" + df_m.loc[i,"도로명주소"]
    lat = df_m.loc[i,"위도"]
    long = df_m.loc[i,"경도"]
    
    folium.CircleMarker([lat, long], tooltip=tooltip).add_to(m)

m
# -

# ### Markers
# ##### There are numerous marker types, starting with a simple Leaflet style location marker with a popup and tooltip HTML.
