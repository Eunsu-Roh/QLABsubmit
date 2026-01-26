"""
Competition data loader
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .preprocessing import validate_quantum_states, to_torch_tensors


class CompetitionDataLoader:
    """
    2nd Quantum AI Competition 데이터 로더
    """
    
    PHASE_NAMES = ['Cluster(SPT)', 'Trivial', 'Ferromagnetic', 'Anti-ferro']
    
    def __init__(self, data_dir='./', batch_size=4, shuffle=True):
        """
        Args:
            data_dir: 데이터 디렉토리
            batch_size: 배치 크기
            shuffle: 데이터 셔플 여부
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_X = None
        self.train_Y = None
        self.train_loader = None
    
    def load_data(self, train_X_file='train_X.npy', train_Y_file='train_y.npy'):
        """
        .npy 파일 로드
        
        Args:
            train_X_file: 입력 데이터 파일명
            train_Y_file: 라벨 데이터 파일명
        
        Returns:
            train_X, train_Y (NumPy arrays)
        """
        print(f"\n{'='*70}")
        print("  데이터 로딩")
        print(f"{'='*70}")
        
        try:
            X_path = os.path.join(self.data_dir, train_X_file)
            Y_path = os.path.join(self.data_dir, train_Y_file)
            
            self.train_X = np.load(X_path)
            self.train_Y = np.load(Y_path)
            
            print(f"[OK] train_X: {self.train_X.shape} ({self.train_X.dtype})")
            print(f"[OK] train_Y: {self.train_Y.shape} ({self.train_Y.dtype})")
            
            # 정규화 검증
            self._validate_data()
            
            # 라벨 분포
            self._print_label_distribution()
            
            return self.train_X, self.train_Y
            
        except FileNotFoundError:
            print(f"\n❌ 파일을 찾을 수 없습니다!")
            print(f"\n다음 명령어로 데이터를 다운로드하세요:")
            print("wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_X.npy")
            print("wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_Y.npy")
            raise
    
    def _validate_data(self):
        """데이터 유효성 검사"""
        print(f"\n[CHECK] 양자 상태 검증:")
        
        is_valid = validate_quantum_states(self.train_X)
        
        if is_valid:
            print("   [OK] 모든 상태가 정규화됨")
        else:
            print("   [WARN] 일부 상태 정규화 필요")
    
    def _print_label_distribution(self):
        """라벨 분포 출력"""
        unique, counts = np.unique(self.train_Y, return_counts=True)
        
        print(f"\n[INFO] 클래스 분포:")
        for label, count in zip(unique, counts):
            phase_name = self.PHASE_NAMES[label]
            print(f"   Label {label} ({phase_name}): {count}개")
    
    def get_dataloader(self):
        """
        PyTorch DataLoader 생성
        
        Returns:
            DataLoader
        """
        if self.train_X is None:
            raise RuntimeError("Call load_data() first")
        
        # NumPy → Torch
        t_train_X, t_train_Y = to_torch_tensors(self.train_X, self.train_Y)
        
        # Dataset & DataLoader
        dataset = TensorDataset(t_train_X, t_train_Y)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        
        print(f"\n✅ DataLoader 생성")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Num batches: {len(self.train_loader)}")
        
        return self.train_loader
    
    def get_torch_tensors(self):
        """
        PyTorch tensors 반환
        
        Returns:
            train_X, train_Y (torch.Tensor)
        """
        if self.train_X is None:
            raise RuntimeError("Call load_data() first")
        
        return to_torch_tensors(self.train_X, self.train_Y)
