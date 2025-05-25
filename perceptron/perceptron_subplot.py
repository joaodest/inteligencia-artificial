import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random

random.seed(42)

class Perceptron:
    def __init__(self, learning_rate=0.1, max_iters=100):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.history = []  # Para armazenar histórico de pesos para visualização
        self.cost_history = []  # Para armazenar histórico de custos
       
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        iters = 0
        
        # Coluna do bias
        X = np.concatenate(
            (X, np.asarray([[1] * X.shape[0]]).T),
            axis=1
        )
        
        # Weights aleatórios
        w = np.random.random(X.shape[1])
        
        # Armazenar pesos iniciais
        self.history.append(w.copy())
        
        # Calcular custo inicial
        initial_cost = self._calculate_cost(X, y, w)
        self.cost_history.append(initial_cost)
        
        for _ in range(self.max_iters):
            has_error = False
            y_pred_all = []
            
            for idx in range(X.shape[0]):
                x_sample, y_sample = X[idx], y[idx]
                y_pred = int(np.sum(w * x_sample) >= 0.5)
                
                if y_pred != y_sample:
                    has_error = True
                    # Usar taxa de aprendizado para ajustes mais suaves
                    if y_pred == 0 and y_sample == 1:
                        w = w + self.learning_rate * x_sample
                    elif y_pred == 1 and y_sample == 0:
                        w = w - self.learning_rate * x_sample
                
                y_pred_all.append(y_pred)
            
            # Armazenar pesos após cada época completa
            self.history.append(w.copy())
            
            # Calcular e armazenar custo
            cost = self._calculate_cost(X, y, w)
            self.cost_history.append(cost)
            
            iters += 1
            if np.equal(np.array(y_pred_all), y).all():
                # Convergiu, não há mais erros
                break
            
            if not has_error:
                # Não houve erros nesta época
                break
            
        self.iters, self.w = iters, w
        print(f"Treinamento concluído em {iters} épocas")
                
    def _calculate_cost(self, X, y, w):
        """Calcula o erro quadrático médio (MSE)"""
        y_pred = (X @ w >= 0.5).astype(int)
        return np.mean((y_pred - y) ** 2)
                
    def predict(self, X):
        X = np.asarray(X)
        X = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        return (X @ self.w >= 0.5).astype(int)
        
    def visualize_cost_function(self, X, y, ax, title='Função de Custo'):
        """
        Visualiza a função de custo para os primeiros dois pesos (w0, w1)
        mantendo o bias (w2) constante.
        """
        X, y = np.asarray(X), np.asarray(y)
        
        # Adicionar coluna de bias
        X_bias = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        
        # Criar grade de valores para w0 e w1
        w0_range = np.linspace(-5, 5, 50)
        w1_range = np.linspace(-5, 5, 50)
        w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
        
        # Calcular custo para cada combinação de w0 e w1
        cost_grid = np.zeros_like(w0_grid)
        
        # Usar o valor final de w2 (bias)
        w2 = self.w[2] if hasattr(self, 'w') else 0
        
        for i in range(len(w0_range)):
            for j in range(len(w1_range)):
                w = np.array([w0_grid[i, j], w1_grid[i, j], w2])
                y_pred = (X_bias @ w >= 0.5).astype(int)
                cost_grid[i, j] = np.mean((y_pred - y) ** 2)
        
        # Plotar superfície de custo
        surf = ax.plot_surface(w0_grid, w1_grid, cost_grid, cmap=cm.viridis,
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Plotar caminho do treinamento se disponível
        if hasattr(self, 'history') and len(self.history) > 0:
            w0_history = [w[0] for w in self.history]
            w1_history = [w[1] for w in self.history]
            cost_history = self.cost_history
            
            # Plotar caminho
            ax.plot(w0_history, w1_history, cost_history, 'r-', linewidth=2, 
                   label='Caminho do Gradiente')
            
            # Marcar pontos inicial e final
            ax.scatter(w0_history[0], w1_history[0], cost_history[0], 
                      color='green', s=100, label='Ponto Inicial')
            ax.scatter(w0_history[-1], w1_history[-1], cost_history[-1], 
                      color='red', s=100, label='Ponto Final')
        
        ax.set_xlabel('Peso w0')
        ax.set_ylabel('Peso w1')
        ax.set_zlabel('Custo (MSE)')
        ax.set_title(title)
        ax.view_init(elev=30, azim=45)
    
    def visualize_training(self, X, y, ax, save_animation=False, filename=None):
        """
        Visualiza a evolução da reta de decisão durante o treinamento
        
        Args:
            X: Dados de entrada
            y: Rótulos
            ax: Eixo matplotlib para plotar
            save_animation: Se True, salva a animação como GIF
            filename: Nome do arquivo para salvar a animação
        """
        if not hasattr(self, 'history') or len(self.history) == 0:
            print("Treine o modelo primeiro usando o método fit()")
            return
        
        # Preparar dados para visualização
        X = np.asarray(X)
        y = np.asarray(y)
        
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
        
        # Inicializar a linha
        line, = ax.plot([], [], 'k-', lw=2, label='Linha de decisão')
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        
        # Plotar a linha de decisão final
        w_final = self.w
        x = np.linspace(min_x, max_x, 100)
        y_final = (0.5 - w_final[0] * x - w_final[2]) / w_final[1]
        ax.plot(x, y_final, 'g--', lw=1.5, label='Linha final')
        
        # Função de inicialização para animação
        def init():
            line.set_data([], [])
            iteration_text.set_text('')
            return line, iteration_text
        
        # Função de atualização para animação
        def update(frame):
            w = self.history[frame]
            
            # Atualizar a linha de decisão
            if w[1] != 0:  # Evitar divisão por zero
                x = np.linspace(min_x, max_x, 100)
                y = (0.5 - w[0] * x - w[2]) / w[1]
                line.set_data(x, y)
            
            iteration_text.set_text(f'Época: {frame}')
            return line, iteration_text
        
        # Criar animação
        ani = FuncAnimation(ax.figure, update, frames=len(self.history),
                            init_func=init, blit=True, interval=500, repeat=True)
        
        ax.set_title(f'Perceptron - Épocas: {self.iters}')
        ax.set_xlabel('Entrada X1')
        ax.set_ylabel('Entrada X2')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True)
        
        if save_animation and filename:
            ani.save(filename, writer='pillow', fps=2)
            print(f"Animação salva como {filename}")
        
        return ani

def visualize_logic_gates():
    """
    Treina e visualiza o perceptron para as portas lógicas AND, OR e XOR em um único subplot
    """
    # Criar figura com subplots para visualização da linha de decisão
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Visualização da Linha de Decisão do Perceptron', fontsize=16)
    
    # Criar figura com subplots para visualização da função de custo
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 10))
    fig2.suptitle('Visualização da Função de Custo do Perceptron', fontsize=16)
    
    # Dados para a porta lógica AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    # Dados para a porta lógica OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # Treinar perceptron para AND
    print("Treinando perceptron para porta lógica AND...")
    perceptron_and = Perceptron(learning_rate=0.01, max_iters=100)
    perceptron_and.fit(X_and, y_and)
    
    # Treinar perceptron para OR
    print("\nTreinando perceptron para porta lógica OR...")
    perceptron_or = Perceptron(learning_rate=0.01, max_iters=100)
    perceptron_or.fit(X_or, y_or)
    
    print("\nTreinando perceptron para porta lógica XOR...")
    perceptron_xor = Perceptron(learning_rate=0.01, max_iters=100)
    perceptron_xor.fit(X_xor, y_xor)
    
    # Visualizar treinamento para AND
    axs1[0].set_title('Porta Lógica AND')
    ani_and = perceptron_and.visualize_training(X_and, y_and, axs1[0], save_animation=True, filename="and_training.gif")
    
    # Visualizar treinamento para OR
    axs1[1].set_title('Porta Lógica OR')
    ani_or = perceptron_or.visualize_training(X_or, y_or, axs1[1], save_animation=True, filename="or_training.gif")
    
    # Visualizar treinamento para XOR
    axs1[2].set_title('Porta Lógica XOR')
    ani_xor = perceptron_xor.visualize_training(X_xor, y_xor, axs1[2], save_animation=True, filename="xor_training.gif")
    
    # Configurar subplots 3D para visualização da função de custo
    for i in range(3):
        axs2[i] = plt.subplot(1, 3, i+1, projection='3d')
    
    # Visualizar função de custo para AND
    perceptron_and.visualize_cost_function(X_and, y_and, axs2[0], title='Função de Custo - AND')
    
    # Visualizar função de custo para OR
    perceptron_or.visualize_cost_function(X_or, y_or, axs2[1], title='Função de Custo - OR')
    
    # Visualizar função de custo para XOR
    perceptron_xor.visualize_cost_function(X_xor, y_xor, axs2[2], title='Função de Custo - XOR')
    # Ajustar layout
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Salvar figuras
    fig1.savefig('perceptron_decision_boundaries.png', dpi=300, bbox_inches='tight')
    fig2.savefig('perceptron_cost_functions.png', dpi=300, bbox_inches='tight')
    print("\nFiguras salvas como 'perceptron_decision_boundaries.png' e 'perceptron_cost_functions.png'")
    
    # Imprimir resultados
    print("\nResultados para AND:")
    for x, y_true in zip(X_and, y_and):
        y_pred = perceptron_and.predict([x])[0]
        print(f"Entrada: {x}, Saída Prevista: {y_pred}, Saída Real: {y_true}")
    
    print("\nResultados para OR:")
    for x, y_true in zip(X_or, y_or):
        y_pred = perceptron_or.predict([x])[0]
        print(f"Entrada: {x}, Saída Prevista: {y_pred}, Saída Real: {y_true}")
        
    print("\nResultados para XOR:")
    for x, y_true in zip(X_xor, y_xor):
        y_pred = perceptron_xor.predict([x])[0]
        print(f"Entrada: {x}, Saída Prevista: {y_pred}, Saída Real: {y_true}")
    
    # Salvar figura final
    plt.savefig('perceptron_logic_gates.png', dpi=300, bbox_inches='tight')
    print("\nFigura salva como 'perceptron_logic_gates.png'")
    
    return (fig1, fig2), (ani_and, ani_or, ani_xor)

if __name__ == "__main__":
    # Treinar e visualizar perceptron para portas lógicas
    figs, animations = visualize_logic_gates()
    
    # Mostrar as figuras
    plt.show()
