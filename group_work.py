import pandas as pd 
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')

df.drop(['id','bdate','has_photo','has_mobile','occupation_name'],axis=1,inplace=True)
df.info()

def sex_apply(sex):
    if sex ==2:
        return 0 
    return 1
df['education_form'].fillna('Full-time',inplace=True)
df['sex'] = df['sex'].apply(sex_apply)
print(df['education_form'].value_counts())
#df.info()

def edu_stat_apply(edu):
    if edu == 'Undergraduate applicant':
        return 0 
    elif edu == "Student (Master's)" or edu == "Student (Bachelor's)" or edu == "Student (Specialist)":
        return 1 
    elif edu == "Alumnus (Specialist) " or edu == "Alumnus (Bachelor's)" or edu == "Alumnus (Master's)":
        return 2 
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_stat_apply)

#df.info()


def lan_apply(lan):
    if lan.find('Русский') != -1:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(lan_apply)
print(df['langs'].value_counts())

print(df['occupation_type'].value_counts())
def occu_typ_apply(ocu):
    if ocu == 'university':
        return 1
    else:
        return 0
df['occupation_type'] = df['occupation_type'].apply(occu_typ_apply)
print(df['occupation_type'].value_counts())
df.info()

#Отделим целевую переменную от осталных данных 

y_sex = df['sex'] #что это?
x_sex= df.drop('sex',axis = 1)#Данные про полы людей


y_edu = df['education_status']
x_edu = df.drop('education_status',axis = 1)


y_lan = df['langs']
x_lan = df.drop('langs',axis = 1)

x_ocu = df.drop('occupation_type',axis = 1)
y_ocu = df['occupation_type']

SEED = 42

X_train,X_test,y_train,y_test = train_test_split(x_sex,y_sex, test_size = 0.25,random_state=SEED)
print(X_train.shape)

#Выполнить стандартизацию показателей в обоих наборах данных ЧЕГОО?
#Стандартицзация значений 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#шаг 3 Содать обьект KNN настроить параметры модели
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)
#шаг 5 рассчитать прогноз значений 
y_pred = classifier.predict(X_test)

#percent = accuracy_score(Y_test,y_pred)*100
#print(percent)
from sklearn.metrics import classification_report, confusion_matrix
a=(classification_report(y_test,y_pred))
b=(confusion_matrix(y_test,y_pred))
print(a)
print(b)
'''
import matplotlib.pyplot as plt 

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')

'''