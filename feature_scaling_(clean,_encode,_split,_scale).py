
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()               #scaler:- variable name &&  StandardScaler() :- method name
X_scaled = scaler.fit_transform()       #X_scaled:- variable name &&  fit_transform() :- method name(It's method need for when you use data spliting from "0 to 1" range)

scaler = MinMaxScaler()                 #scaler:- variable name &&  MinMaxScaler()  :- method name
X_scaled = scaler.fit_transform()       #X_scaled:- variable name &&  fit_transform() :- method name(It's method need for when you use data spliting from "0 to 1" range)


















import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

data = {
    "StudyHours" : [1, 2, 3, 4, 5],
    "TestScore" : [55, 65, 75, 85, 95]
}

df = pd.DataFrame(data)





#standardScaler
standard_scaler = StandardScaler()                    #standard_scaler:- variable name && StandardScaler():- methode name
standard_scaled = standard_scaler.fit_transform(df)   #fit_transform(df):- using this methode To send input as a (df)
                                                      #fit_transform:- It's method every columns to find "Mean" & "StandardScaler" tham generate output for 0 to 1 range
print("_____Standard Scaler Output_____")
print(pd.DataFrame(standard_scaled, columns=['StudyHours', 'TestScore']))





#minMaxScaler
minMax_scaler = MinMaxScaler()                        #minMax_scaler:- variable name && MinMaxScaler():- methode name
minMax_scaled = minMax_scaler.fit_transform(df)       #fit_transform(df):- using this methode To send input as a (df)
                                                      #fit_transform:- It's method every columns to learn than Rescale for range between 0 to 1
print("\n\n_____Min Max Scaler Output_____")
print(pd.DataFrame(minMax_scaled, columns=['StudyHours', 'TestScore']))





#Train_Test_Split                                     #IMPORTANT :- When you this portion work with than always use [[]] "in add columns name"
X = df[["StudyHours"]]                                #X = df[["StudyHours"]] :- as a Input data
y = df[["TestScore"]]                                 #y = df[["TestScore"]] :-  as a Output data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)         #random_state=42 :- to generate consistent Result

print("\n\nTraining Data : ")
print(X_train)                                        #Output explaination:- as(1,2,3,4 [80%] used for training purpose)
print(y_train)                                        #Output explaination:- as(55,75,85,95 [80%] used for training purpose)
print("\n\nTesting Data : ")
print(X_test)                                         #Output explaination:- as(5 [20%] used for testing purpose)
print(y_test)                                         #Output explaination:- as(95 [20%] used for testing purpose)