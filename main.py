# Здесь должен быть твой код
import pandas as pd 
jup = pd.read_csv('titanic.csv')

jup.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace = True)

jup.drop(['Embarked'],axis = 1,inplace = True)

age_1 = jup[jup['Pclass']==1]['Age'].median()
age_2 = jup[jup['Pclass']==2]['Age'].median()    
age_3 = jup[jup['Pclass']==3]['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        if row['Pclass'] == 2:
            return age_2
  
        return age_3
    return row['Age']
jup['Age'] = jup.apply(fill_age,axis = 1)

def fill_sex(sex):
    if sex == 'male':
        return 1 
    return 0 


jup['Sex'] = jup['Sex'].apply(fill_sex)
jup.info()







y = jup['Sex']
Xr = jup.drop('Sex',axis =1)#Данные о поссажирах
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(Xr,y,test_szie = 0.20)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print(percent = accuracy_score(y_test,y_pred) * 100)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))