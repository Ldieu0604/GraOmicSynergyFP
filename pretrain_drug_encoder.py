import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_scatter import scatter_mean 
from mol_bpe import Tokenizer
from utils import TestbedDataset

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=5):
        super(GINEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        nn_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn_mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            nn_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn_mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x

# ==========================================
# 1. KHỐI PRE-TRAINER & HÀM LOSS INFONCE (FRAGPOOL)
# ==========================================
class DualViewPretrainer(nn.Module):
    def __init__(self, atom_in_dim=78, frag_in_dim=2300, hidden_dim=128, proj_dim=64):
        super(DualViewPretrainer, self).__init__()
        self.atom_encoder = GINEncoder(in_dim=atom_in_dim, hidden_dim=hidden_dim, num_layers=5)
        self.frag_encoder = GINEncoder(in_dim=frag_in_dim, hidden_dim=hidden_dim, num_layers=2)
        
        # Projection Head
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.frag_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    # ĐÃ SỬA ĐẦU VÀO: Bỏ batch_atom/batch_frag, thêm frag_atom_mapping
    def forward(self, x_atom, edge_index_atom, x_frag, edge_index_frag, frag_atom_mapping):
        # 1. Trích xuất đặc trưng Cấp độ Node (Node-level)
        h_atom = self.atom_encoder(x_atom, edge_index_atom) # [tổng_số_nguyên_tử, hidden_dim]
        h_frag = self.frag_encoder(x_frag, edge_index_frag) # [tổng_số_mảnh, hidden_dim]
        
        # 2. CƠ CHẾ FRAGPOOL
        # Lấy index tương ứng từ ma trận mapping
        frag_indices = frag_atom_mapping[0]  # Index của Mảnh
        atom_indices = frag_atom_mapping[1]  # Index của Nguyên tử
        
        # Rút trích các vector nguyên tử
        atom_embs = h_atom[atom_indices]
        
        # Gom cụm (Averaging) các nguyên tử thành mảnh tương ứng
        # dim_size đảm bảo chiều dài tensor đầu ra bằng đúng tổng số mảnh
        h_A = scatter_mean(atom_embs, frag_indices, dim=0, dim_size=h_frag.size(0))
        
        # 3. Phóng qua Projection Head để tạo không gian đối chiếu
        return self.atom_proj(h_A), self.frag_proj(h_frag)

def contrastive_loss_infoNCE(z_atom, z_frag, temperature=0.1):
    # Chuẩn hóa L2
    z_atom = F.normalize(z_atom, dim=1)
    z_frag = F.normalize(z_frag, dim=1)
    
    # Tính ma trận tương đồng Cosine
    logits = torch.matmul(z_atom, z_frag.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    # Tính Cross Entropy cả 2 chiều
    loss_a2f = F.cross_entropy(logits, labels)
    loss_f2a = F.cross_entropy(logits.T, labels)
    return (loss_a2f + loss_f2a) / 2

# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def run_pretraining(epochs=50, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 1. Khởi tạo mô hình
    try:
        tokenizer = Tokenizer("vocab.txt")
        FRAG_VOCAB_SIZE = tokenizer.num_subgraph_type()
        print(f"Đã nhận diện bộ từ vựng với kích thước (frag_in_dim): {FRAG_VOCAB_SIZE}")
    except Exception as e:
        print("Cảnh báo: Không tải được vocab.txt, dùng kích thước mặc định 2300.")
        FRAG_VOCAB_SIZE = 2300
        
    model = DualViewPretrainer(atom_in_dim=78, frag_in_dim=FRAG_VOCAB_SIZE, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Đang khởi tạo TestbedDataset...")
    dataset = TestbedDataset(root="data/split_data/all_test", dataset="GDSC_train_dc")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Bắt đầu Pre-training InfoNCE (với cơ chế FRAGPOOL)...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Khai thác CẢ Thuốc A và Thuốc B
            # -- Xử lý Thuốc A --
            # Truyền frag_atom_mapping_1 vào thay cho batch_atom / batch_frag
            out_atom_A, out_frag_A = model(batch.x_1, batch.edge_index_1, 
                                           batch.x_f1, batch.edge_index_f1, 
                                           batch.frag_atom_mapping_1)
            loss_A = contrastive_loss_infoNCE(out_atom_A, out_frag_A)
            
            # -- Xử lý Thuốc B --
            out_atom_B, out_frag_B = model(batch.x_2, batch.edge_index_2, 
                                           batch.x_f2, batch.edge_index_f2, 
                                           batch.frag_atom_mapping_2)
            loss_B = contrastive_loss_infoNCE(out_atom_B, out_frag_B)
            
            # Tổng hợp mất mát
            loss = (loss_A + loss_B) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Average InfoNCE Loss: {avg_loss:.4f}")
        
    # 3. Lưu trọng số
    os.makedirs("pretrained_weights", exist_ok=True)
    torch.save(model.atom_encoder.state_dict(), "pretrained_weights/pretrained_atom_encoder.pt")
    torch.save(model.frag_encoder.state_dict(), "pretrained_weights/pretrained_frag_encoder.pt")
    print("Đã lưu trọng số Pre-trained thành công tại thư mục 'pretrained_weights'!")

if __name__ == "__main__":
    run_pretraining()