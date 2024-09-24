import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def treinar_mlp(x, y, arquitetura, arquivo_nome, max_iter, ativacao, solver):
    media_exec = [] 
    desvio_padrao_exec = [] 

    for i in range(10):  
        print(f'Treinando RNA - Execução {i+1} para o arquivo {arquivo_nome}')
        regr = MLPRegressor(hidden_layer_sizes=arquitetura,
                            max_iter=max_iter,
                            activation=ativacao,
                            solver=solver,
                            learning_rate='adaptive',
                            n_iter_no_change=50)

        regr.fit(x, y)
        y_est = regr.predict(x)
        
        erro = np.mean((y - y_est) ** 2)  
        media_exec.append(erro)

        
        plt.figure(figsize=[14, 7])

        plt.subplot(1, 3, 1)
        plt.plot(x, y, label='Dados Originais')
        plt.title('Dados Originais')
        
        plt.subplot(1, 3, 2)
        plt.plot(regr.loss_curve_, label='Curva de Aprendizado')
        plt.title('Curva de Aprendizado')

        plt.subplot(1, 3, 3)
        plt.plot(x, y, linewidth=1, color='yellow', label='Dados Originais')
        plt.plot(x, y_est, linewidth=2, label='Predição')
        plt.title('Regressão')
        plt.legend()
        
        
        resultados_dir = 'C:\\Users\\(RandomUser)\\Downloads\\RNA\\resultados'
        os.makedirs(resultados_dir, exist_ok=True)
        
        # Salva a figura
        plt.savefig(f'{resultados_dir}\\{arquivo_nome}_arquitetura_{arquitetura}_exec_{i+1}.png')
        plt.close()  

    return np.mean(media_exec), np.std(media_exec)

def carregar_dados(arquivo_path):
    arquivo = np.load(arquivo_path)
    x = arquivo[0]
    y = np.ravel(arquivo[1])
    return x, y


def executar_testes_para_arquivos():
    arquivos = ['teste2.npy', 'teste3.npy', 'teste4.npy', 'teste5.npy']
    arquiteturas = [(10, 10), (20, 20, 20), (50, 25)]

    for arquivo_nome in arquivos:
        x, y = carregar_dados(f'C:\\Users\\(RandomUser)\\Downloads\\RNA\\{arquivo_nome}')
        for arquitetura in arquiteturas:
            media_exec, desvio_padrao_exec = treinar_mlp(x, y, arquitetura, arquivo_nome, 10000, 'relu', 'adam')
            print(f'Média do erro: {media_exec:.4f}, Desvio Padrão do erro: {desvio_padrao_exec:.4f}')


executar_testes_para_arquivos()
