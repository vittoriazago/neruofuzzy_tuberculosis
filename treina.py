import pandas as pd
df = pd.read_csv('diagnostico.csv')
X_df = df[['febre', 'tosse','sangue_escarro','dor_toracica','perda_peso','tuberculina','historico','opacidade_raiox']]
Y_df = df['diagnostico']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-10:]
teste_marcacoes = Y[-10:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
print("Resultado: " + str(resultado))

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print("Acertos: " + str(taxa_de_acerto) + "%")
print("Total de elementos testados: " + str(total_de_elementos))