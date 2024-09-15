## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
### STEP 1:
Read the given Data.
### STEP 2:
Clean the Data Set using Data Cleaning Process.
### STEP 3:
Apply Feature Encoding for the feature in the data set.
### STEP 4:
Apply Feature Transformation for the feature in the data set.
### STEP 5:
Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
### Developed by: HARISH S
### REG-NO: 212223230071
      
```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![319860534-7518dfdd-a869-4481-adf5-15b7d77a2299](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/54cb1733-92b4-4ae4-8c26-39b22fb601a3)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![319860584-d2b0a391-66ad-42eb-bd6a-989a53dc1463](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/4486794d-fa4c-4b5e-9d3f-9bb487935edb)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![319860639-f44e225e-8c22-4c00-ad38-961facb3caa7](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/73d145a6-7952-4628-a1c4-0a9b2d4b9c63)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![319860684-08c9ad56-23ba-4549-ada7-185105d50f26](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/84ace74d-5506-4271-b597-a57b39404ce0)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![319860943-8ed1f84c-e7db-485d-958a-1751672b3955](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/dfe7db86-cc6c-435e-a2ec-f9077c249a77)

```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![319860994-8b001581-9300-40bd-97dc-4a434c93307f](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/996ad06d-4eab-4c8b-8698-7e20adf526ee)

```py
pd.get_dummies(df2,columns=["nom_0"])
```

![319861036-5a7a6ad1-4703-4778-aacb-2efb0f2849cb](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/1cd50575-594e-472e-ad38-53748017df3c)


```py
pip install --upgrade category_encoders
```
![319861068-090b221f-1cfa-4d83-9aca-d6fcc438cd5b](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/6280d03d-738d-4562-8cff-86ec391b3080)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![319861102-b76dd80d-bef7-4b62-8077-26fee40db525](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/79f7bd03-143f-41bb-9ba9-d887c5047a1d)

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![319861131-52f84e42-82d1-43a0-a2ce-d4843c5ae781](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/c793e894-6ba2-4b58-8a74-5719eac26ae0)

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![319861160-261246fd-6ab1-4942-a95f-f803b0a976a8](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/77958cf3-4fcc-4948-88d0-1fa76aa542b9)

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![319861181-cca33cc1-c50b-4618-a6de-b9236dd26507](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/01945111-85d7-45df-8d14-28353790ee38)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![319861204-b44233ab-f18f-4222-b63f-29a74702838a](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/c9dd7ac6-2559-48fa-990c-0f1d8343c2cd)

```py
df.skew()
```

![319861240-3429a0d5-1d3d-4660-b79b-46a2b12a8050](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/e6eee121-4382-4c0b-92a5-ca2abc95dba4)

```py
np.log(df["Highly Positive Skew"])
```

![319861274-d4fbd14d-0f73-4810-9a37-7f90e09c7f0d](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/1f6efbeb-d82c-4c44-8313-357592c03b3a)

```py
np.reciprocal(df["Moderate Positive Skew"])
```

![319861571-62102da6-8d20-47c9-8aaa-f5543409c8c8](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/793e6cd0-51bc-410d-85c4-9e308fcc80c2)

```py
np.sqrt(df["Highly Positive Skew"])
```

![319861624-186d2b92-d488-41a6-a178-0df95e0c8364](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/75735f09-adc0-4593-a51d-7f8cc6d23181)

```py
np.square(df["Highly Positive Skew"])
```

![319861686-5336a561-434d-4552-944a-97535ccbcc29](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/df7e24f8-f1a6-4a18-a70d-867055969eda)

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![319861757-fd614327-c87a-46fa-883e-0dce5e136cce](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/2532f626-ec68-4ec9-b417-6e6d98e79f57)

```py
df.skew()
```

![319861793-805e86cf-f9f5-4e20-a2ea-6b7328bc6443](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/8409f0fd-1134-41f1-9d54-6c2dc904edc2)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![319861825-6ce34d13-c097-4706-b409-4cf80b55e39e](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/66035a7a-5708-430e-a2a3-0c0191dee49b)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![319861866-55399f4f-9c8c-4be0-b747-31da55de4387](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/87966f02-982b-4c43-bdf0-b9ee9992dbbe)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![319861899-54adc02b-d562-442c-be3a-466037a3c753](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/4f686d45-2277-46ac-a498-241f17a297f3)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![319861945-0388b0c8-b211-45ad-a901-321712a8c922](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/31e9fb93-4a26-45b3-a4c0-1a230b83b9ac)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![319861982-c82b4630-bfe9-4d2b-9250-bb22d423b2f4](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/633826a7-3484-481b-a03b-4ca033021613)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![319862041-e86811e2-d338-4d47-8285-9387cf87c191](https://github.com/Praveen0500/EXNO-3-DS/assets/120218611/0a23c373-4be7-4f31-ac5a-cb697781b896)

# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
