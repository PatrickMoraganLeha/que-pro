import pandas as pd
df = pd.read_csv('titanic.csv')
#print(df.groupby('Sex')['Survived'].mean())
#print(df.pivot_table(index = 'Survived',columns = 'Pclass',values = 'Age', aggfunc = 'mean'))
(df.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace = True))#удаляем их на
print(df['Embarked'].value_counts())#запалняем значиние порт
df['Embarked'].fillna('S',inplace = True)
print(df.groupby('Pclass')['Age'].median())#запалняем занчение возростами 
#вычесляем медианный возраст пассажиров и заполняем пустыми значениеми 
#выстовляем пассажиров в разные классы
age_1 = df[df['Pclass']==1]['Age'].median()
age_2 = df[df['Pclass']==2]['Age'].median()
age_3 = df[df['Pclass']==3]['Age'].median()

#создаем функцию каторая будет сортировать в классы возраст 1 класса меньше 30 и т.д
def fill_age(row):#row - это Series канткретного пассажира 
    # Тело функции
df['Age'] = df.apply(fill_age, axis = 1)


