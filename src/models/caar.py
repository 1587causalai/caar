"""
基于推断/行动(Abduction/Action)的新型回归模型 (CAAR)

该模型通过推断网络(Abduction Network)为每个样本推断潜在子群体的柯西分布，
然后通过行动网络(Action Network)定义从潜在表征到结果的映射规则。
利用柯西分布的特性实现端到端学习，天然对异常值具有鲁棒性。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

class AbductionNetwork(nn.Module):
    """
    推断网络(Abduction Network)
    
    对于每个输入特征x_i，推断其在潜在表征空间中对应的"子群体"或"影响区域"的柯西分布参数。
    输出描述潜在子群体U_i的柯西分布的参数：位置l_i和尺度s_i，两者维度均为latent_dim。
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        """
        初始化推断网络
        
        参数:
            input_dim: 输入特征维度
            latent_dim: 潜在表征空间维度 (l_i 和 s_i 各自的维度)
            hidden_dims: 共享的MLP部分的隐藏层维度列表。此MLP的输出维度为 hidden_dims[-1]。
        """
        super(AbductionNetwork, self).__init__()
        
        # 构建共享的MLP层，用于特征提取
        # 这个MLP的输出维度将是 hidden_dims 的最后一个元素
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim_i in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim_i # 更新 prev_dim 为当前隐藏层维度
        
        # 位置参数l_i的输出网络 (在共享MLP之后)
        # 输入维度是共享MLP的输出维度 prev_dim (即 hidden_dims[-1])
        # 输出维度是 latent_dim
        self.loc_net = nn.Sequential(
            *shared_layers,
            nn.Linear(prev_dim, latent_dim) 
        )
        
        # 尺度参数s_i的输出网络 (在共享MLP之后)
        # 注意：这里复用了 shared_layers 的定义，意味着 loc_net 和 scale_net 共享了前面的MLP层。
        # 也可以选择为 scale_net 构建独立的 shared_layers_scale，如果希望它们不共享特征提取器。
        # 当前实现是共享的，这在许多多头架构中很常见。
        self.scale_net = nn.Sequential(
            *shared_layers, # 复用相同的共享层实例
            nn.Linear(prev_dim, latent_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            l_i: 位置参数 [batch_size, latent_dim]
            s_i: 尺度参数 [batch_size, latent_dim]
        """
        # 计算位置参数
        l_i = self.loc_net(x)
        
        # 计算尺度参数，使用softplus确保尺度为正
        s_i = F.softplus(self.scale_net(x))
        
        return l_i, s_i

class ActionNetwork(nn.Module):
    """
    行动网络(Action Network)
    
    定义从任何潜在表征u到最终结果y的确定性映射规则。
    采用简单的共享线性层: y(u) = w^T u + b
    """
    def __init__(self, latent_dim):
        """
        初始化行动网络
        
        参数:
            latent_dim: 潜在表征空间维度
        """
        super(ActionNetwork, self).__init__()
        
        # 线性映射层
        self.linear = nn.Linear(latent_dim, 1)
    
    def forward(self, u):
        """
        前向传播
        
        参数:
            u: 潜在表征 [batch_size, latent_dim]
            
        返回:
            y: 预测结果 [batch_size, 1]
        """
        return self.linear(u)
    
    def get_weights(self):
        """
        获取权重和偏置
        
        返回:
            w: 权重向量
            b: 偏置
        """
        return self.linear.weight.data.squeeze(), self.linear.bias.data.item()

class CAAR(nn.Module):
    """
    基于推断/行动的新型回归模型 (CAAR: Cauchy Abduction Action Regression)
    
    结合推断网络和行动网络，利用柯西分布的特性实现端到端学习。
    推断网络 AbductionNetwork 输出潜在柯西分布的参数 (l_i, s_i)，维度均为 latent_dim。
    行动网络 ActionNetwork (一个线性层) 从 l_i (维度 latent_dim) 预测 y。
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[128, 64]):
        """
        初始化CAAR模型
        
        参数:
            input_dim: 输入特征维度
            latent_dim: 潜在表征空间维度 (l_i, s_i 各自的维度, 也是 ActionNetwork 的输入维度)
            hidden_dims: AbductionNetwork 内部共享MLP的隐藏层维度列表。
                         AbductionNetwork 内部共享MLP的输出维度是 hidden_dims[-1]。
                         然后 loc_net 和 scale_net 将此 hidden_dims[-1] 维的表示映射到 latent_dim 维。
                         如果 hidden_dims[-1] == latent_dim，则 loc_net/scale_net 的最后线性层是 Identity-like (如果参数允许)。
        """
        super(CAAR, self).__init__()
        
        # 初始化推断网络
        self.abduction_net = AbductionNetwork(input_dim, latent_dim, hidden_dims)
        
        # 初始化行动网络
        self.action_net = ActionNetwork(latent_dim)
        
        # 记录模型配置
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            mu_y: 预测的y的位置参数 (点估计) [batch_size, 1]
            gamma_y: 预测的y的尺度参数 (与预测相关的不确定性或离散度) [batch_size, 1]
        """
        # 1. 推断网络 (Abduction Network):
        #   输入 x (input_dim)
        #   内部共享MLP (由 hidden_dims 定义) 将 x 映射到一个中间表示 (维度 hidden_dims[-1])
        #   然后 loc_net 和 scale_net 分别将此中间表示映射为：
        #     l_i (位置参数, latent_dim 维)
        #     s_i (尺度参数, latent_dim 维, 经过 softplus 保证为正)
        l_i, s_i = self.abduction_net(x)
        
        # 2. 行动网络 (Action Network):
        #   获取行动网络 (一个线性层 nn.Linear(latent_dim, 1)) 的权重 w (latent_dim) 和偏置 b (标量)
        w, b = self.action_net.get_weights()
        
        # 3. 计算最终预测 y 的柯西分布参数:
        #   预测的y的位置参数 (mu_y) 是 l_i 通过行动网络得到的线性变换。
        #   mu_y = w^T * l_i + b
        mu_y = torch.matmul(l_i, w.unsqueeze(1)) + b
        
        #   预测的y的尺度参数 (gamma_y) 是 s_i 通过行动网络权重的绝对值得到的线性变换。
        #   gamma_y = |w|^T * s_i
        #   这种方式保证了 gamma_y > 0 (因为 s_i > 0 且 |w| >= 0)，并且允许不同潜在维度的贡献是可加的。
        w_abs = torch.abs(w)
        gamma_y = torch.matmul(s_i, w_abs.unsqueeze(1))
        
        return mu_y, gamma_y
    
    def predict(self, x):
        """
        预测函数，返回y的位置参数作为点估计
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            mu_y: 预测的y的位置参数 [batch_size, 1]
        """
        mu_y, _ = self.forward(x)
        return mu_y
    
    def loss_function(self, y_true, mu_y, gamma_y):
        """
        计算负对数似然损失
        
        参数:
            y_true: 真实标签 [batch_size, 1]
            mu_y: 预测的y的位置参数 [batch_size, 1]
            gamma_y: 预测的y的尺度参数 [batch_size, 1]
            
        返回:
            loss: 负对数似然损失
        """
        # 柯西分布的负对数似然
        # L = -sum_i log[1/(pi*gamma_y_i * (1 + ((y_i - mu_y_i)/gamma_y_i)^2))]
        
        # 计算标准化残差: (y - mu) / gamma
        normalized_residuals = (y_true - mu_y) / gamma_y
        
        # 计算柯西分布的负对数似然
        cauchy_nll = torch.log(np.pi * gamma_y) + torch.log(1 + normalized_residuals**2)
        
        # 返回平均损失
        return torch.mean(cauchy_nll)

class CAARModel:
    """
    CAAR模型的包装类，提供训练和预测接口
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        """
        初始化CAAR模型
        
        参数:
            input_dim: 输入特征维度
            latent_dim: 潜在表征空间维度
            hidden_dims: 隐藏层维度列表
            lr: 学习率
            batch_size: 批量大小
            epochs: 训练轮数
            device: 训练设备
            early_stopping_patience: 早停的耐心轮数。如果为None或0，则不启用早停。
            early_stopping_min_delta: 认为验证损失有显著改善的最小变化量。
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = CAAR(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_time': 0,
            'best_epoch': 0 
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        else: # 如果没有验证集，则早停无效
            self.early_stopping_patience = None 
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                mu_y, gamma_y = self.model(batch_X)
                loss = self.model.loss_function(batch_y, mu_y, gamma_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf') # Default if no validation
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val, gamma_y_val = self.model(X_val_tensor)
                    current_val_loss = self.model.loss_function(y_val_tensor, mu_y_val, gamma_y_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'CAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                # 早停和模型检查点逻辑
                if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy() # 保存最佳模型状态
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.')
                        break # 中断训练循环
            else: # 没有验证集的情况
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'CAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
        
        self.history['train_time'] = time.time() - start_time

        # 如果启用了早停并且找到了更好的模型状态，则加载最佳模型权重
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'Loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'CAAR Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {epoch+1}')
        return self
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 测试集特征
            
        返回:
            y_pred: 预测结果
        """
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        # 预测
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        
        return y_pred.flatten()
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
            params: 模型参数字典
        """
        return {
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
    
    def set_params(self, **params):
        """
        设置模型参数，并重新初始化模型和优化器
        """
        # 更新存储的参数
        self.__init__(**params) # type: ignore
        return self

class MLP(nn.Module):
    """
    标准的多层感知机 (MLP) 用于回归
    """
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        """
        初始化MLP模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表。MLP主干网络的输出维度是 hidden_dims[-1]。
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1)) # 输出层，预测单个回归值
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim # 保存input_dim
        self.hidden_dims = hidden_dims # 保存hidden_dims

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            y_pred: 预测结果 [batch_size, 1]
        """
        return self.network(x)

class MLPModel:
    """
    MLP模型的包装类，提供训练和预测接口，使用MSE损失
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = MLP(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_time': 0,
            'best_epoch': 0
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        
        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        else:
            self.early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        final_epoch = self.epochs

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                y_pred = self.model(batch_X)
                loss = self.loss_fn(y_pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    current_val_loss = self.loss_fn(y_val_pred, y_val_tensor).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP at epoch {epoch+1}.')
                        final_epoch = epoch + 1
                        break 
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
            final_epoch = epoch + 1 # Update in case loop finishes normally

        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'MLP loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'MLP Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self):
        return {
            'input_dim': self.model.input_dim,
            'hidden_dims': self.model.hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def set_params(self, **params):
        if 'lr' in params:
            self.lr = params['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'epochs' in params:
            self.epochs = params['epochs']
        # Note: Changing hidden_dims or input_dim would require re-initializing the model
        return self

class GaussianAbductionNetwork(nn.Module):
    """
    高斯推断网络 (Gaussian Abduction Network)
    对于每个输入特征x_i，推断其在潜在表征空间中对应的"子群体"的高斯分布参数。
    输出描述潜在子群体U_i的高斯分布的参数：均值 mu_z 和标准差 sigma_z (通过softplus保证为正)，
    两者维度均为latent_dim。
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        super(GaussianAbductionNetwork, self).__init__()
        # 构建共享的MLP层，用于特征提取
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim_i in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim_i
        
        # 均值 mu_z 的输出网络 (在共享MLP之后)
        self.mu_net = nn.Sequential(
            *shared_layers,
            nn.Linear(prev_dim, latent_dim)
        )
        
        # 标准差 sigma_z 的输出网络 (在共享MLP之后)
        # 使用与 mu_net 相同的共享特征提取层
        self.sigma_net = nn.Sequential(
            *shared_layers, 
            nn.Linear(prev_dim, latent_dim)
        )

    def forward(self, x):
        mu_z = self.mu_net(x)
        # 计算尺度参数 sigma_z，使用softplus确保尺度为正
        sigma_z = F.softplus(self.sigma_net(x)) 
        return mu_z, sigma_z

class GAAR(nn.Module):
    """
    高斯推断行动回归 (GAAR: Gaussian Abduction Action Regression)
    推断网络 GaussianAbductionNetwork 输出潜在高斯分布的参数 (mu_z, sigma_z)。
    行动网络 ActionNetwork (一个线性层) 从 mu_z 预测 y 的均值 mu_y。
    输出y的尺度 sigma_y 也基于 sigma_z 和行动网络权重推导。
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[128, 64]):
        super(GAAR, self).__init__()
        self.gaussian_abduction_net = GaussianAbductionNetwork(input_dim, latent_dim, hidden_dims)
        self.action_net = ActionNetwork(latent_dim) 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    def forward(self, x):
        mu_z, sigma_z = self.gaussian_abduction_net(x) # sigma_z 已经是正的

        w, b = self.action_net.get_weights()
        
        mu_y = torch.matmul(mu_z, w.unsqueeze(1)) + b
        
        var_y = torch.sum((w**2) * (sigma_z**2), dim=1, keepdim=True)
        sigma_y = torch.sqrt(var_y + 1e-6) 
        
        return mu_y, sigma_y

    def predict(self, x):
        mu_y, _ = self.forward(x)
        return mu_y

    def loss_function(self, y_true, mu_y, sigma_y):
        """
        高斯负对数似然损失 (忽略常数项)
        NLL = 0.5 * sum_i [ log(sigma_y_i^2) + ((y_i - mu_y_i)/sigma_y_i)^2 ]
        """
        # sigma_y 已经通过softplus和sqrt(var_y + eps)保证接近正值
        # 但为防止极小sigma_y导致log(0)或除以0，仍可加epsilon
        sigma_y_stable = sigma_y + 1e-8 
        log_var_y = torch.log(sigma_y_stable**2) # log(sigma^2) = 2 * log(sigma)
        squared_error_term = ((y_true - mu_y) / sigma_y_stable)**2
        gaussian_nll = 0.5 * (log_var_y + squared_error_term)
        return torch.mean(gaussian_nll)

class GAARModel:
    """
    GAAR模型的包装类
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[128, 64],
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = GAAR(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        else:
            self.early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        start_time = time.time()
        final_epoch = self.epochs

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                mu_y, sigma_y = self.model(batch_X)
                loss = self.model.loss_function(batch_y, mu_y, sigma_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val, sigma_y_val = self.model(X_val_tensor)
                    current_val_loss = self.model.loss_function(y_val_tensor, mu_y_val, sigma_y_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'GAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for GAAR at epoch {epoch+1}.')
                        final_epoch = epoch + 1
                        break
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'GAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
            final_epoch = epoch + 1 # Update in case loop finishes normally
            
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'GAAR loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'GAAR Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self):
        return {
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def set_params(self, **params):
        if 'lr' in params:
            self.lr = params['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'epochs' in params:
            self.epochs = params['epochs']
        return self

def pinball_loss(y_true, y_pred, quantile=0.5):
    """
    Pinball Loss function for Quantile Regression.

    Parameters:
        y_true: True values (Tensor)
        y_pred: Predicted values (Tensor)
        quantile: The quantile to be predicted (float between 0 and 1)
    
    Returns:
        loss: Tensor, the mean pinball loss
    """
    error = y_true - y_pred
    loss = torch.mean(torch.max((quantile - 1) * error, quantile * error))
    return loss

class MLPPinball(nn.Module):
    """
    标准的多层感知机 (MLP) 用于分位数回归 (使用Pinball Loss)
    结构与普通MLP相同，但设计用于配合Pinball Loss进行训练。
    """
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(MLPPinball, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1)) # 输出层，预测单个分位数的值
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim 
        self.hidden_dims = hidden_dims

    def forward(self, x):
        return self.network(x)

class MLPPinballModel:
    """
    MLP Pinball模型的包装类，提供训练和预测接口，使用Pinball Loss
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], quantile=0.5, 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = MLPPinball(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.quantile = quantile 
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_time': 0,
            'best_epoch': 0
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        else:
            self.early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        final_epoch = self.epochs
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                y_pred = self.model(batch_X)
                loss = pinball_loss(batch_y, y_pred, self.quantile) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    current_val_loss = pinball_loss(y_val_tensor, y_val_pred, self.quantile).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Pinball Epoch {epoch+1}/{self.epochs}, Train Pinball Loss: {avg_train_loss:.4f}, Val Pinball Loss: {current_val_loss:.4f}')

                if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP Pinball at epoch {epoch+1}.')
                        final_epoch = epoch + 1
                        break
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Pinball Epoch {epoch+1}/{self.epochs}, Train Pinball Loss: {avg_train_loss:.4f}')
            final_epoch = epoch + 1 # Update in case loop finishes normally
            
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'MLP Pinball loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'MLP Pinball Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self):
        return {
            'input_dim': self.model.input_dim,
            'hidden_dims': self.model.hidden_dims,
            'quantile': self.quantile,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def set_params(self, **params):
        if 'quantile' in params:
            self.quantile = params['quantile']
        if 'lr' in params:
            self.lr = params['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'epochs' in params:
            self.epochs = params['epochs']
        return self

# Huber Loss Function (can be replaced by nn.HuberLoss if preferred)
def huber_loss_fn(y_true, y_pred, delta=1.0):
    """
    Computes the Huber loss.

    Parameters:
        y_true: Ground truth values.
        y_pred: Predicted values.
        delta: The point where the Huber loss changes from a quadratic to linear.

    Returns:
        The Huber loss.
    """
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)

class MLPHuber(nn.Module):
    """
    MLP for Huber Regression. Structure is identical to the standard MLP.
    The Huber loss will be applied in the wrapper model.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(MLPHuber, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1)) # Output a single value for regression
        
        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim # Store for potential re-initialization or checks

    def forward(self, x):
        return self.model(x)

class MLPHuberModel:
    """
    Wrapper for the MLPHuber network, providing a scikit-learn-like interface.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], delta=1.0,
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        """
        Initializes the MLPHuberModel.

        Args:
            input_dim: Dimension of the input features.
            hidden_dims: List of dimensions for hidden layers.
            delta: The delta parameter for the Huber loss (epsilon in sklearn's HuberRegressor).
            lr: Learning rate.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            device: Device to train on ('cuda' or 'cpu').
            early_stopping_patience: Number of epochs to wait for improvement before stopping.
            early_stopping_min_delta: Minimum change in validation loss to be considered an improvement.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.delta = delta
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        self.model = MLPHuber(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None
        
        start_training_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            permutation = torch.randperm(X_train_tensor.size()[0])
            
            epoch_train_loss = 0.0
            num_batches = 0

            for i in range(0, X_train_tensor.size()[0], self.batch_size):
                self.optimizer.zero_grad()
                indices = permutation[i:i+self.batch_size]
                batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                
                outputs = self.model(batch_X)
                loss = huber_loss_fn(batch_y, outputs, self.delta) # Use Huber loss
                
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                num_batches +=1
            
            avg_epoch_train_loss = epoch_train_loss / num_batches

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = huber_loss_fn(y_val_tensor, val_outputs, self.delta).item() # Use Huber loss
                
                if verbose and (epoch + 1) % 10 == 0 :
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
                    if val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_model_state = self.model.state_dict() 
                    else:
                        epochs_no_improve += 1
                    
                    if epochs_no_improve >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}')
                        if best_model_state is not None: # Load best model
                            self.model.load_state_dict(best_model_state)
                        break 
            elif verbose and (epoch + 1) % 10 == 0:
                 print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_epoch_train_loss:.4f}')
        
        # If early stopping wasn't triggered or not enabled, but we did track best_model_state (because X_val was provided)
        # ensure the best model is loaded if it was better than the final epoch's model.
        if X_val is not None and y_val is not None and hasattr(self, 'early_stopping_patience') and self.early_stopping_patience is not None and self.early_stopping_patience > 0 and best_model_state is not None:
             # Check if the loop finished naturally (not by break) but a better model was found
            if epochs_no_improve < self.early_stopping_patience and val_loss >= best_val_loss : # type: ignore
                 self.model.load_state_dict(best_model_state)
                 if verbose:
                    print(f'Finished training. Loaded best model state with val loss: {best_val_loss:.4f}')


        end_training_time = time.time()
        if verbose:
            print(f"Training finished. Total time: {end_training_time - start_training_time:.2f}s")


    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

    def get_params(self):
        """获取模型参数，用于sklearn兼容性 (例如网格搜索)"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'delta': self.delta,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }

    def set_params(self, **params):
        """
        设置模型参数，并重新初始化模型和优化器
        兼容sklearn的set_params，主要用于网格搜索等场景。
        """
        # 更新所有可设置的参数
        for param, value in params.items():
            setattr(self, param, value)
        
        # 重新初始化模型和优化器以反映新参数
        # 注意: input_dim 通常不应在 set_params 中更改，因为它通常由数据集决定。
        # 如果确实需要更改，需要确保外部代码正确处理了。
        self.model = MLPHuber(self.input_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return self
