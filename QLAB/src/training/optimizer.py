"""
Training loop and optimizer utilities
"""

import torch
import torch.optim as optim
from .loss import get_loss_function


def create_optimizer(model, optimizer_name='adam', lr=0.05, **kwargs):
    """
    Optimizer 생성
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'adamw', 'sgd', 'rmsprop', 'lbfgs'
        lr: learning rate
        **kwargs: optimizer별 추가 인자
    
    Returns:
        optimizer
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'lbfgs':
        return optim.LBFGS(model.parameters(), lr=lr, max_iter=20, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, train_loader, optimizer, loss_fn, device='cpu'):
    """
    한 에폭 훈련
    
    Args:
        model: VQCClassifier
        train_loader: DataLoader
        optimizer: PyTorch optimizer
        loss_fn: 손실 함수
        device: 'cpu' or 'cuda'
    
    Returns:
        avg_loss, avg_accuracy
    """
    model.train()
    
    # L-BFGS는 특별한 처리 필요
    if isinstance(optimizer, optim.LBFGS):
        return train_epoch_lbfgs(model, train_loader, optimizer, loss_fn, device)
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        # Forward
        optimizer.zero_grad()
        predictions = model(batch_X)
        
        # Loss
        loss = loss_fn(predictions, batch_Y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred_classes = torch.argmax(predictions, dim=1)
        correct += (pred_classes == batch_Y).sum().item()
        total += batch_Y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def train_epoch_lbfgs(model, train_loader, optimizer, loss_fn, device='cpu'):
    """
    L-BFGS용 한 에폭 훈련 (closure function 필요)
    
    Args:
        model: VQCClassifier
        train_loader: DataLoader
        optimizer: L-BFGS optimizer
        loss_fn: 손실 함수
        device: 'cpu' or 'cuda'
    
    Returns:
        avg_loss, avg_accuracy
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        def closure():
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_Y)
            loss.backward()
            return loss
        
        # L-BFGS step with closure
        loss = optimizer.step(closure)
        
        # Metrics
        with torch.no_grad():
            predictions = model(batch_X)
            total_loss += loss.item()
            pred_classes = torch.argmax(predictions, dim=1)
            correct += (pred_classes == batch_Y).sum().item()
            total += batch_Y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def train_model(
    model,
    train_loader,
    epochs=200,
    lr=0.05,
    loss_name='ce',
    optimizer_name='adam',
    device='cpu',
    verbose=True,
    scheduler_name=None,
    weight_decay=0.0,
    **loss_kwargs
):
    """
    전체 훈련 프로세스
    
    Args:
        model: VQCClassifier
        train_loader: DataLoader
        epochs: 에폭 수
        lr: learning rate
        loss_name: 손실 함수 ('ce', 'focal')
        optimizer_name: optimizer ('adam', 'sgd')
        device: 'cpu' or 'cuda'
        verbose: 진행상황 출력 여부
        scheduler_name: 'cosine' or None
        weight_decay: L2 regularization factor
        **loss_kwargs: 손실 함수 추가 인자 (class_weights 등)
    
    Returns:
        loss_history, acc_history, best_params
    """
    if verbose:
        print(f"\n{'='*70}")
        print("  훈련 시작")
        print(f"{'='*70}")
        print(f"   에폭: {epochs}")
        print(f"   학습률: {lr}")
        print(f"   손실 함수: {loss_name}")
        if loss_kwargs.get('class_weights') is not None:
            print(f"   클래스 가중치: {loss_kwargs['class_weights'].tolist()}")
        print(f"   Optimizer: {optimizer_name}")
        print(f"   Device: {device}")
        if scheduler_name:
            print(f"   Scheduler: {scheduler_name}")
        if weight_decay > 0:
            print(f"   Weight Decay: {weight_decay}")
    
    model.to(device)
    optimizer = create_optimizer(model, optimizer_name, lr, weight_decay=weight_decay)
    loss_fn = get_loss_function(loss_name, **loss_kwargs)
    
    scheduler = None
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    loss_history = []
    acc_history = []
    best_acc = 0.0
    best_params = None
    
    if verbose:
        print(f"\n{'Epoch':>5} {'Loss':>10} {'Accuracy':>10}")
        print("-" * 30)
    
    for epoch in range(epochs):
        avg_loss, avg_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        
        if scheduler:
            scheduler.step()
        
        # Best model 저장
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = model.params.detach().clone()
        
        # 진행상황 출력
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"{epoch+1:5d} {avg_loss:10.4f} {avg_acc:10.4f}")
    
    if verbose:
        print("-" * 30)
        print(f"✅ 훈련 완료!")
        print(f"   최종 Loss: {loss_history[-1]:.4f}")
        print(f"   최종 Accuracy: {acc_history[-1]:.4f}")
        print(f"   최고 Accuracy: {best_acc:.4f}")
    
    # Best params 복원
    if best_params is not None:
        model.params.data = best_params
    
    return loss_history, acc_history, best_params
