import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from tqdm import tqdm

random.seed(42)

class MLP:
    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = []
        self.layers = [2, 3, 1]  # Default: 2 entradas, 3 neurônios na camada oculta, 1 saída
        self.loss = []
        self.weight_history = []  # Para armazenar histórico de pesos para visualização
        self.momentum = 0.9

        
    def _loss_function(self, y_true, y_pred):
        # Erro quadrático médio
        return np.mean((y_true - y_pred) ** 2)
    
    def _derivative_loss(self, y_true, y_pred):
        # Derivada do erro quadrático médio
        return -2 * (y_true - y_pred) / y_true.shape[1]
    
    def _sigmoid(self, z):
        # Função sigmoide
        return 1 / (1 + np.exp(-z))
    
    def _derivative_sigmoid(self, z):
        # Derivada da função sigmoide
        sigmoid_z = self._sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    def _forward_pass(self, X):
        input_to_layer = np.copy(X)
        activations = [input_to_layer]
        derivatives = [np.zeros(X.shape)]
        z_values = []  # Para armazenar os valores z para visualização
        
        # Camada oculta e camadas de saída
        for i in range(len(self.weights)):
            z_i = np.matmul(self.weights[i], input_to_layer) + self.bias[i]
            input_to_layer = self._sigmoid(z_i)
            activations.append(input_to_layer)
            derivatives.append(self._derivative_sigmoid(z_i))
            z_values.append(z_i)
        
        return activations, derivatives, z_values
    
    def _backward_pass(self, activations, derivatives, y):
        # Calcular e armazenar a perda
        current_loss = self._loss_function(y, activations[-1])
        self.loss.append(current_loss)
        
        # Inicializar gradientes
        dl_dw = [None] * len(self.weights)
        dl_db = [None] * len(self.bias)
        
        # Calcular gradientes para a camada de saída
        dl_dy_last = self._derivative_loss(y, activations[-1])
        dl_dz_last = np.multiply(dl_dy_last, derivatives[-1])
        dl_dw[-1] = np.matmul(dl_dz_last, activations[-2].T)
        dl_db[-1] = np.sum(dl_dz_last, axis=1, keepdims=True)
        
        # Propagar o erro para as camadas anteriores
        delta = dl_dz_last
        
        # Camadas ocultas
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.multiply(np.matmul(self.weights[i+1].T, delta), derivatives[i+1])
            dl_dw[i] = np.matmul(delta, activations[i].T)
            dl_db[i] = np.sum(delta, axis=1, keepdims=True)
        
        return dl_dw, dl_db
        
    def _update_weights(self, dl_dw, dl_db):
        # Atualizar pesos e bias usando gradiente descendente
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + self.lr * dl_dw[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.lr * dl_db[i]
            self.weights[i] -= self.velocity_w[i]
            self.bias[i] -= self.velocity_b[i]
        
        # Armazenar pesos para visualização
        self.weight_history.append([np.copy(w) for w in self.weights])
            
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Garantir que y tem a forma correta (n_samples, n_outputs)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Transpor X para ter forma (n_features, n_samples)
        X = X.T
        y = y.T
        
        # Inicializar pesos e bias
        self.weights = []
        self.bias = []
        self.loss = []
        self.weight_history = []
        
        # Inicializar pesos para cada camada com valores pequenos aleatórios
        for idx in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[idx+1], self.layers[idx]) * np.sqrt(1 / self.layers[idx]))
            self.bias.append(np.random.randn(self.layers[idx+1], 1) * np.sqrt(1 / self.layers[idx]))
    
        # Inicializar arrays de velocidade para momentum
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.bias]
        
        # Armazenar pesos iniciais
        self.weight_history.append([np.copy(w) for w in self.weights])
        
        # Treinar o modelo
        pbar = tqdm(range(self.epochs), desc="Treinando", leave=True)
        for epoch in pbar:
            # Forward pass
            activations, derivatives, _ = self._forward_pass(X)
            
            # Backward pass
            dl_dw, dl_db = self._backward_pass(activations, derivatives, y)
            
            # Atualizar pesos
            self._update_weights(dl_dw, dl_db)
            
            # Atualizar barra de progresso
            if len(self.loss) > 0:
                pbar.set_postfix({"loss": f"{self.loss[-1]:.6f}"})
            
            # Verificar convergência
            if len(self.loss) > 1 and abs(self.loss[-1] - self.loss[-2]) < 1e-6 and self.loss[-1] < 0.01:
                pbar.set_description(f"Convergência atingida após {epoch} épocas")
                break
            
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        # Garantir que X tem a forma correta
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Transpor X para ter forma (n_features, n_samples)
        X = X.T
        
        # Forward pass
        activations, _, _ = self._forward_pass(X)
        
        # Retornar a saída da última camada
        return activations[-1].T
    
    def visualize_decision_boundary(self, X, y, ax, epoch=None, save_animation=False, filename=None):
        """
        Visualiza a fronteira de decisão da MLP
        
        Args:
            X: Dados de entrada
            y: Rótulos
            ax: Eixo matplotlib para plotar
            epoch: Época específica para visualizar
            save_animation: Se True, salva a animação como GIF
            filename: Nome do arquivo para salvar a animação
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.flatten()
        
        # Plotar os pontos de dados
        classes = np.unique(y)
        colors = ['blue', 'red', 'green', 'purple']
        markers = ['o', 's', '^', 'D']
        
        for i, label in enumerate(classes):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      label=f'Classe {label}', 
                      color=colors[i % len(colors)],
                      marker=markers[i % len(markers)],
                      s=100, 
                      alpha=0.7)
        
        # Definir limites do gráfico
        min_x, max_x = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        min_y, max_y = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Criar grade para plotar a fronteira de decisão
        xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100),
                             np.linspace(min_y, max_y, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Usar pesos específicos para a época se fornecidos
        if epoch is not None and epoch < len(self.weight_history):
            original_weights = self.weights.copy()
            original_bias = self.bias.copy()
            
            # Usar pesos da época específica
            self.weights = self.weight_history[epoch]
        
        # Calcular predições na grade
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Restaurar pesos originais se necessário
        if epoch is not None and epoch < len(self.weight_history):
            self.weights = original_weights
            self.bias = original_bias
        
        # Plotar fronteira de decisão
        contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
        
        ax.set_xlabel('Entrada X1')
        ax.set_ylabel('Entrada X2')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True)
        
        return contour
    
    def visualize_training_animation(self, X, y, ax, save_animation=False, filename=None):
        """
        Cria uma animação mostrando a evolução da fronteira de decisão durante o treinamento
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.flatten()
        
        # Plotar pontos de dados
        classes = np.unique(y)
        colors = ['blue', 'red', 'green', 'purple']
        markers = ['o', 's', '^', 'D']
        
        for i, label in enumerate(classes):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      label=f'Classe {label}', 
                      color=colors[i % len(colors)],
                      marker=markers[i % len(markers)],
                      s=100, 
                      alpha=0.7)
        
        # Definir limites do gráfico
        min_x, max_x = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        min_y, max_y = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Criar grade para visualização da fronteira
        xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100),
                           np.linspace(min_y, max_y, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Inicialização da fronteira
        contour = [ax.contourf(xx, yy, np.zeros_like(xx), levels=[0, 0.5, 1], 
                             alpha=0.3, colors=['blue', 'red'])]
        
        # Texto para mostrar a época e perda
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        
        # Função de inicialização para animação
        def init():
            contour[0].remove()
            contour[0] = ax.contourf(xx, yy, np.zeros_like(xx), levels=[0, 0.5, 1], 
                                   alpha=0.3, colors=['blue', 'red'])
            epoch_text.set_text('')
            return contour + [epoch_text]
        
        # Função de atualização para animação
        def update(frame):
            # Remover contorno anterior
            contour[0].remove()
            
            # Guardar pesos originais
            original_weights = self.weights.copy()
            original_bias = self.bias.copy()
            
            # Usar pesos da época específica
            self.weights = self.weight_history[frame]
            
            # Calcular predições na grade
            Z = self.predict(grid_points)
            Z = Z.reshape(xx.shape)
            
            # Atualizar contorno
            contour[0] = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], 
                                   alpha=0.3, colors=['blue', 'red'])
            
            # Restaurar pesos originais
            self.weights = original_weights
            self.bias = original_bias
            
            # Atualizar texto
            if frame < len(self.loss):
                epoch_text.set_text(f'Época: {frame}, Perda: {self.loss[frame]:.6f}')
            else:
                epoch_text.set_text(f'Época: {frame}')
            
            return contour + [epoch_text]
        
        # Criar animação
        # Selecionar frames uniformemente distribuídos para mostrar a evolução da rede
        if len(self.weight_history) > 20:
            # Pegar mais frames no início e no final do treinamento para mostrar a convergência
            num_frames = 30  # Total de frames desejados
            # Pegamos 1/3 dos frames do início, 1/3 uniformemente distribuídos no meio e 1/3 do final
            start_frames = 10
            end_frames = 10
            middle_frames = num_frames - start_frames - end_frames
            
            # Índices dos frames
            start_indices = list(range(start_frames))
            end_indices = list(range(len(self.weight_history) - end_frames, len(self.weight_history)))
            
            # Índices do meio uniformemente distribuídos
            if middle_frames > 0 and len(self.weight_history) > (start_frames + end_frames):
                step = (len(self.weight_history) - start_frames - end_frames) / (middle_frames + 1)
                middle_indices = [int(start_frames + i * step) for i in range(1, middle_frames + 1)]
            else:
                middle_indices = []
            
            # Combinar todos os índices
            frame_indices = start_indices + middle_indices + end_indices
        else:
            frame_indices = range(len(self.weight_history))
        
        ani = FuncAnimation(ax.figure, update, frames=frame_indices,
                          init_func=init, blit=False, interval=150, repeat=True)
        
        ax.set_title(f'MLP - Épocas: {len(self.weight_history)-1}')
        ax.set_xlabel('Entrada X1')
        ax.set_ylabel('Entrada X2')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True)
        
        if save_animation and filename:
            try:
                print(f"Salvando animação como {filename}...")
                # Usar um número reduzido de frames para evitar problemas de memória
                ani.save(filename, writer='pillow', fps=2, dpi=80)
                print(f"Animação salva como {filename}")
            except Exception as e:
                print(f"Erro ao salvar animação: {e}")
                # Fallback: apenas gerar uma imagem estática
                plt.savefig(filename.replace('.gif', '.png'))
                print(f"Imagem estática salva como {filename.replace('.gif', '.png')}")
        
        return ani
    
    def visualize_loss(self, ax):
        """
        Visualiza a curva de perda durante o treinamento
        """
        ax.plot(self.loss)
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.set_title('Curva de Perda')
        ax.grid(True)

def visualize_logic_gates():
    """
    Treina e visualiza a MLP para as portas lógicas AND, OR e XOR
    """
    # Criar pasta para salvar as visualizações se não existir
    import os
    output_dir = "./mlp_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Criar figura com subplots para visualização da fronteira de decisão
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Visualização da Fronteira de Decisão da MLP', fontsize=16)
    
    # Criar figura com subplots para visualização da função de perda
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Visualização da Função de Perda da MLP', fontsize=16)
    
    # Dados para a porta lógica AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([[0], [0], [0], [1]])
    
    # Dados para a porta lógica OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([[0], [1], [1], [1]])
    
    # Dados para a porta lógica XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Treinar MLP para AND
    print("\nTreinando MLP para porta lógica AND...")
    mlp_and = MLP(lr=0.5, epochs=1000)
    mlp_and.layers = [2, 3, 1]  # 2 entradas, 3 neurônios na camada oculta, 1 saída
    mlp_and.fit(X_and, y_and)
    
    # Treinar MLP para OR
    print("\nTreinando MLP para porta lógica OR...")
    mlp_or = MLP(lr=0.5, epochs=1000)
    mlp_or.layers = [2, 3, 1]
    mlp_or.fit(X_or, y_or)
    
    # Treinar MLP para XOR
    print("\nTreinando MLP para porta lógica XOR...")
    mlp_xor = MLP(lr=0.5, epochs=2000)  # Taxa de aprendizado menor e mais épocas
    mlp_xor.layers = [2, 5, 5, 1]  # Aumentando para 4 neurônios na camada oculta
    mlp_xor.fit(X_xor, y_xor)
    
    # Visualizar treinamento para AND
    axs1[0].set_title('Porta Lógica AND')
    print("\nGerando animação para AND...")
    ani_and = mlp_and.visualize_training_animation(X_and, y_and, axs1[0], 
                                                 save_animation=True, 
                                                 filename="./mlp_output/mlp_and_training.gif")
    
    # Visualizar treinamento para OR
    axs1[1].set_title('Porta Lógica OR')
    print("\nGerando animação para OR...")
    ani_or = mlp_or.visualize_training_animation(X_or, y_or, axs1[1], 
                                               save_animation=True, 
                                               filename="./mlp_output/mlp_or_training.gif")
    
    # Visualizar treinamento para XOR
    axs1[2].set_title('Porta Lógica XOR')
    print("\nGerando animação para XOR...")
    ani_xor = mlp_xor.visualize_training_animation(X_xor, y_xor, axs1[2], 
                                                 save_animation=True, 
                                                 filename="./mlp_output/mlp_xor_training.gif")
    
    # Criar visualização estática das fronteiras de decisão para AND, OR e XOR em uma única figura
    print("\nGerando visualizações estáticas para AND, OR e XOR...")
    fig_static = plt.figure(figsize=(18, 6))
    
    # Visualização AND
    ax_and = fig_static.add_subplot(1, 3, 1)
    mlp_and.visualize_decision_boundary(X_and, y_and, ax_and)
    ax_and.set_title('Fronteira de Decisão - AND')
    
    # Visualização OR
    ax_or = fig_static.add_subplot(1, 3, 2)
    mlp_or.visualize_decision_boundary(X_or, y_or, ax_or)
    ax_or.set_title('Fronteira de Decisão - OR')
    
    # Visualização XOR
    ax_xor = fig_static.add_subplot(1, 3, 3)
    mlp_xor.visualize_decision_boundary(X_xor, y_xor, ax_xor)
    ax_xor.set_title('Fronteira de Decisão - XOR')
    
    # Ajustar layout e salvar a figura
    fig_static.tight_layout()
    fig_static.savefig('./mlp_output/mlp_static_boundaries.png', dpi=300, bbox_inches='tight')
    
    # Visualizar curvas de perda
    mlp_and.visualize_loss(axs2[0])
    axs2[0].set_title('Curva de Perda - AND')
    
    mlp_or.visualize_loss(axs2[1])
    axs2[1].set_title('Curva de Perda - OR')
    
    mlp_xor.visualize_loss(axs2[2])
    axs2[2].set_title('Curva de Perda - XOR')
    
    # Ajustar layout
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Salvar figuras
    fig1.savefig('./mlp_output/mlp_decision_boundaries.png', dpi=300, bbox_inches='tight')
    # Salvar figura final
    plt.savefig('./mlp_output/mlp_logic_gates.png', dpi=300, bbox_inches='tight')
    print("\nFigura salva como './mlp_output/mlp_logic_gates.png'")
    
    # Imprimir resultados
    print("\nResultados para AND:")
    for x, y_true in zip(X_and, y_and):
        y_pred = mlp_and.predict(x.reshape(1, -1))
        print(f"Entrada: {x}, Saída Prevista: {y_pred[0][0]:.4f}, Saída Real: {y_true[0]}")
    
    print("\nResultados para OR:")
    for x, y_true in zip(X_or, y_or):
        y_pred = mlp_or.predict(x.reshape(1, -1))
        print(f"Entrada: {x}, Saída Prevista: {y_pred[0][0]:.4f}, Saída Real: {y_true[0]}")
        
    print("\nResultados para XOR:")
    for x, y_true in zip(X_xor, y_xor):
        y_pred = mlp_xor.predict(x.reshape(1, -1))
        print(f"Entrada: {x}, Saída Prevista: {y_pred[0][0]:.4f}, Saída Real: {y_true[0]}")
    
    return (fig1, fig2), (ani_and, ani_or, ani_xor)

if __name__ == "__main__":
    # Treinar e visualizar MLP para portas lógicas
    figs, animations = visualize_logic_gates()
    
    print("\nTodas as visualizações foram salvas na pasta './mlp_output/'")
    
    # Mostrar as figuras
    plt.show()
