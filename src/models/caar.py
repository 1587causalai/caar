"""
基于推断/行动(Abduction/Action)的新型回归模型 (CAAR)

该模型通过推断网络(Abduction Network)为每个样本推断潜在子群体的柯西分布，
然后通过行动网络(Action Network)定义从潜在表征到结果的映射规则。
利用柯西分布的特性实现端到端学习，天然对异常值具有鲁棒性。

统一架构说明:
所有模型（CAAR, GAAR, MLP及其变体）都共享一个完全相同的 UnifiedRegressionNetwork。
这个 UnifiedRegressionNetwork 内部包含: FeatureNetwork -> AbductionNetwork -> ActionNetwork。
- FeatureNetwork: 提取输入特征的高级表征 (representation)。
- AbductionNetwork: 基于 representation 推断一组通用的潜在参数（location_param, scale_param）。
- ActionNetwork: 从 location_param 生成点预测 mu_y。

UnifiedRegressionNetwork.forward() 会输出 (mu_y, location_param, scale_param)。
模型的差异完全由其包装类中的 compute_loss 方法如何解释和利用这些输出来决定。
- CAAR/GAAR: 其 compute_loss 方法利用 mu_y, location_param, scale_param (并结合 ActionNetwork 权重) 计算各自的NLL损失。
- MLP及其变体: 其 compute_loss 方法仅利用 mu_y 进行损失计算 (如MSE, Huber等)，忽略 location_param 和 scale_param (除了 mu_y 本身由 location_param 生成外)。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

class FeatureNetwork(nn.Module):
    """
    特征网络 (Feature Network)
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim_i in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim_i))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim_i
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims 

    def forward(self, x):
        return self.network(x)

class AbductionNetwork(nn.Module):
    """
    统一推断网络 (Abduction Network)
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        super(AbductionNetwork, self).__init__()
        shared_layers_list = []
        prev_dim = representation_dim
        for hidden_dim_i in hidden_dims:
            shared_layers_list.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers_list.append(nn.ReLU())
            prev_dim = hidden_dim_i
        shared_output_dim = prev_dim 
        self.location_head = nn.Linear(shared_output_dim, latent_dim)
        self.scale_head = nn.Linear(shared_output_dim, latent_dim)
        self.shared_mlp = nn.Sequential(*shared_layers_list)
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
            
    def forward(self, representation):
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))
        return location_param, scale_param

class ActionNetwork(nn.Module):
    """
    行动网络(Action Network)
    """
    def __init__(self, latent_dim):
        super(ActionNetwork, self).__init__()
        self.linear = nn.Linear(latent_dim, 1)
        self.latent_dim = latent_dim 
    
    def forward(self, location_param):
        return self.linear(location_param)
    
    def get_weights(self):
        # Ensure weight is 1D for matmul with (batch, latent_dim) in scale calculation for CAAR/GAAR
        # and bias is a scalar.
        weight = self.linear.weight.data
        if weight.ndim > 1:
            weight = weight.squeeze()
        bias = self.linear.bias.data
        if bias.ndim > 0:
            bias = bias.squeeze()
        return weight, bias
        
class UnifiedRegressionNetwork(nn.Module):
    """
    统一回归网络 (Unified Regression Network)
    包含 FeatureNetwork -> AbductionNetwork -> ActionNetwork
    """
    def __init__(self, input_dim, representation_dim, latent_dim, 
                 feature_hidden_dims, abduction_hidden_dims):
        super(UnifiedRegressionNetwork, self).__init__()
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        self.action_net = ActionNetwork(latent_dim)
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims

    def forward(self, x):
        representation = self.feature_net(x)
        location_param, scale_param = self.abduction_net(representation)
        mu_y = self.action_net(location_param)
        return mu_y, location_param, scale_param

    def predict(self, x):
        mu_y, _, _ = self.forward(x)
        return mu_y

class CAARModel:
    """
    CAAR模型的包装类，使用 UnifiedRegressionNetwork
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim, 
                                          self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # For CAAR, location_param is l_i, scale_param is s_i
        # mu_y_pred is already w^T l_i + b
        w, _ = self.model.action_net.get_weights() 
        w_abs = torch.abs(w)
        # Ensure s_i (scale_param) and w_abs are compatible for matmul
        # s_i is [batch_size, latent_dim], w_abs is [latent_dim]
        gamma_y = torch.matmul(scale_param, w_abs.unsqueeze(1)) # Result: [batch_size, 1]
        
        gamma_y_stable = torch.clamp(gamma_y, min=1e-6)
        normalized_residuals = (y_true - mu_y_pred) / gamma_y_stable
        # L = log(pi*gamma) + log(1 + ((y-mu)/gamma)^2)
        cauchy_nll = torch.log(torch.tensor(np.pi, device=mu_y_pred.device) * gamma_y_stable) + torch.log(1 + normalized_residuals**2)
        return torch.mean(cauchy_nll)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'CAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss.')
                        break 
            else: 
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'CAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
        
        self.history['train_time'] = time.time() - start_time

        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'Loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'CAAR Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch_count}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()
    
    def get_params(self, deep=True): 
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str, 
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params
    
    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in model_param_keys:
                 model_params_to_reinit[key] = value
            if key == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']

        self._setup_model_optimizer() 
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self

class MLPModel:
    """
    MLP模型的包装类，使用 UnifiedRegressionNetwork，MSE损失
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64],
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        self._setup_model_optimizer()
        # self.loss_fn = nn.MSELoss() # Will be computed directly in compute_loss
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}

    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim, 
                                          self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # MLP uses MSE, only needs mu_y_pred (derived from location_param via ActionNetwork)
        # location_param and scale_param are ignored here.
        return F.mse_loss(mu_y_pred, y_true)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs
        
        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()

        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP at epoch {epoch+1}.')
                        break 
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')

        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'MLP loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'MLP Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch_count}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self, deep=True):
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params

    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in model_param_keys:
                 model_params_to_reinit[key] = value
            if key == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']

        self._setup_model_optimizer()
        # self.loss_fn = nn.MSELoss() # Re-ensure if it was an attribute, now directly in compute_loss
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self

class GAARModel:
    """
    GAAR模型的包装类 - 使用 UnifiedRegressionNetwork
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64],
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}

    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim, 
                                          self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # For GAAR, location_param is mu_z, scale_param is sigma_z
        # mu_y_pred is already w^T mu_z + b
        w, _ = self.model.action_net.get_weights()
        
        # Ensure w is 1D for w**2, and sigma_z (scale_param) is [batch_size, latent_dim]
        # w is [latent_dim], sigma_z is [batch_size, latent_dim]
        var_y = torch.sum((w.unsqueeze(0)**2) * (scale_param**2), dim=1, keepdim=True) # w broadcasts over batch
        sigma_y = torch.sqrt(var_y + 1e-6)
        
        sigma_y_stable = torch.clamp(sigma_y, min=1e-8)
        # NLL = 0.5 * [log(sigma_y^2) + ((y - mu_y)/sigma_y)^2]
        # log(sigma_y_stable**2) is 2 * log(sigma_y_stable)
        log_var_y = 2 * torch.log(sigma_y_stable) 
        squared_error_term = ((y_true - mu_y_pred) / sigma_y_stable)**2
        gaussian_nll = 0.5 * (log_var_y + squared_error_term)
        return torch.mean(gaussian_nll)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        start_time = time.time()

        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'GAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for GAAR at epoch {epoch+1}.')
                        break
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'GAAR Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
            
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'GAAR loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'GAAR Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch_count}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self, deep=True):
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params

    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            if param in model_param_keys:
                 model_params_to_reinit[param] = value
            if param == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']

        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self

class MLPPinballModel:
    """
    MLP Pinball模型的包装类 - 使用 UnifiedRegressionNetwork
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64],
                 quantile=0.5, 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.quantile = quantile
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim, 
                                self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # Pinball loss only needs mu_y_pred
        error = y_true - mu_y_pred
        loss = torch.max((self.quantile - 1) * error, self.quantile * error)
        return torch.mean(loss)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Pinball Epoch {epoch+1}/{self.epochs}, Train Pinball Loss: {avg_train_loss:.4f}, Val Pinball Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP Pinball at epoch {epoch+1}.')
                        break
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Pinball Epoch {epoch+1}/{self.epochs}, Train Pinball Loss: {avg_train_loss:.4f}')
            
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'MLP Pinball loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'MLP Pinball Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch_count}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self, deep=True):
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'quantile': self.quantile,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params

    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            if param in model_param_keys:
                 model_params_to_reinit[param] = value
            if param == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self

class MLPHuberModel:
    """
    Wrapper for the MLPHuber network - 使用 UnifiedRegressionNetwork
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64],
                 delta=1.0,
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.delta = delta
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}

    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim,
                              self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # Huber loss only needs mu_y_pred
        error = y_true - mu_y_pred
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0 
        best_model_state_dict = None 
        final_epoch_count = self.epochs

        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()

        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                self.optimizer.zero_grad()
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)

            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Huber Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP Huber at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}')
                        break 
            elif verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                 print(f'MLP Huber Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
        
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                 print(f'MLP Huber loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f"MLP Huber Training finished. Total time: {self.history['train_time']:.2f}s. Final epoch: {final_epoch_count}")
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model.predict(X_tensor)
        return predictions.cpu().numpy().flatten()

    def get_params(self, deep=True):
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'delta': self.delta,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params

    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            if param in model_param_keys:
                 model_params_to_reinit[param] = value
            if param == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']

        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self

class MLPCauchyModel:
    """
    Wrapper for the MLPCauchy network, using simplified Cauchy loss - UnifiedRegressionNetwork
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64],
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        self._setup_model_optimizer()
        # self.loss_fn = cauchy_loss_fn # Will be computed directly in compute_loss

        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}

    def _setup_model_optimizer(self):
        self.model = UnifiedRegressionNetwork(self.input_dim, self.representation_dim, self.latent_dim,
                               self.feature_hidden_dims, self.abduction_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_loss(self, y_true, mu_y_pred, location_param, scale_param):
        # Simplified Cauchy loss (log(1 + err^2)), only needs mu_y_pred
        error_sq = (y_true - mu_y_pred)**2
        loss = torch.log(1 + error_sq) 
        return torch.mean(loss)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs
        
        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                mu_y_pred, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, mu_y_pred, location_param, scale_param)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    mu_y_val_pred, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, mu_y_val_pred, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Cauchy Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}')

                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= effective_early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered for MLP Cauchy at epoch {epoch+1}.')
                        break 
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'MLP Cauchy Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}')
            
        self.history['train_time'] = time.time() - start_time
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'MLP Cauchy loaded best model state from epoch {self.history["best_epoch"]} with validation loss: {best_val_loss:.4f}')
        
        if verbose:
            print(f'MLP Cauchy Training completed in {self.history["train_time"]:.2f} seconds. Final epoch: {final_epoch_count}')
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.predict(X_tensor).cpu().numpy()
        return y_pred.flatten()

    def get_params(self, deep=True):
        params = {
            'input_dim': self.model.input_dim,
            'representation_dim': self.model.representation_dim,
            'latent_dim': self.model.latent_dim,
            'feature_hidden_dims': self.model.feature_hidden_dims,
            'abduction_hidden_dims': self.model.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }
        return params

    def set_params(self, **params):
        model_param_keys = ['input_dim', 'representation_dim', 'latent_dim', 'feature_hidden_dims', 'abduction_hidden_dims']
        model_params_to_reinit = {}

        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            if param in model_param_keys:
                 model_params_to_reinit[param] = value
            if param == 'device':
                self.device = value if isinstance(value, torch.device) else torch.device(value)
                self.device_str = str(self.device)
        
        if 'input_dim' in model_params_to_reinit: self.input_dim = model_params_to_reinit['input_dim']
        if 'representation_dim' in model_params_to_reinit: self.representation_dim = model_params_to_reinit['representation_dim']
        if 'latent_dim' in model_params_to_reinit: self.latent_dim = model_params_to_reinit['latent_dim']
        if 'feature_hidden_dims' in model_params_to_reinit: self.feature_hidden_dims = model_params_to_reinit['feature_hidden_dims']
        if 'abduction_hidden_dims' in model_params_to_reinit: self.abduction_hidden_dims = model_params_to_reinit['abduction_hidden_dims']
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
        return self
