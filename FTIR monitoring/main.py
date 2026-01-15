import data_split
import preprocessing
import train_test
import model

path = '/Users/naveenj/Library/CloudStorage/OneDrive-IITDelhi/[01] Rashmi/FTIR_030124_2.xlsx'
target = 'Refolding'
sheet_name = 'Sheet1'
df, y = data_split.split(path, target, sheet_name)
n = len(df)
x = preprocessing.preprocess(df,n)
ts = 0.2
X_train, X_test, y_train, y_test = train_test.tts(x,y,ts)
X_train_scaled, X_test_scaled = preprocessing.norm(X_train, X_test, y_train)
v1, v2, bestp = model.model(X_train_scaled, X_test_scaled, y_train, y_test)

print (v1)
print (v2)
print (bestp)