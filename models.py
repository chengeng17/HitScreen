import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
from coattention import CoAttentionLayer
from torch.nn.utils.weight_norm import weight_norm


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class HitScreen(nn.Module):
    def __init__(self, args):
        super(HitScreen, self).__init__()

        # Initialize parameters from the `args`
        self.is_LigandGCN = args.is_LigandGCN
        self.is_unimol_Ligand = args.is_unimol_Ligand
        self.is_is_ProteinCNN = args.is_ProteinCNN
        self.is_esm_Protein = args.is_esm_Protein

        # Protein extractor (CNN)
        self.protein_extractor = ProteinCNN(
            args.protein_embedding_dim, 
            args.protein_num_filters, 
            args.protein_kernel_size, 
            args.protein_padding, 
            args.is_ProteinCNN
        )

        # Drug Extractor and CoAttention Layer
        if self.is_LigandGCN and self.is_esm_Protein:
            self.drug_extractor = MolecularGCN(
                in_feats=args.ligand_node_in_feats, 
                dim_embedding=args.hidden_size, 
                hidden_feats=args.ligand_hidden_layers
            )
            self.gcn_bcn = CoAttentionLayer(
                v_dim=args.gcn_ligand_node_in_embedding, 
                pLMs_q_dim=args.protein_language_model_embedding_dim, 
                h_dim=args.hidden_size, 
                layer=args.layer
            )
        elif self.is_LigandGCN and self.is_ProteinCNN:
            self.drug_extractor = MolecularGCN(
                in_feats=args.ligand_node_in_feats, 
                dim_embedding=args.hidden_size, 
                hidden_feats=args.ligand_hidden_layers
            )
            self.gcn_bcn = CoAttentionLayer(
                v_dim=args.gcn_ligand_node_in_embedding, 
                esm_q_dim=args.protein_embedding_dim, 
                h_dim=args.hidden_size, 
                layer=args.layer
            )

        # Unimol ligand and Protein co-attention layer
        if self.is_unimol_Ligand and self.is_esm_Protein:
            self.bcn = CoAttentionLayer(
                v_dim=args.unimol_ligand_node_in_embedding, 
                pLMs_q_dim=args.protein_language_model_embedding_dim, 
                h_dim=args.hidden_size, 
                layer=args.layer
            )
        elif self.is_unimol_Ligand and self.is_ProteinCNN:
            self.bcn = CoAttentionLayer(
                v_dim=args.unimol_ligand_node_in_embedding, 
                esm_q_dim=args.protein_embedding_dim, 
                h_dim=args.hidden_size, 
                layer=args.layer
            )

        # MLP Classifier
        self.mlp_classifier = MLPDecoder(
            args.decoder_in_dim, 
            args.decoder_hidden_dim, 
            args.decoder_out_dim, 
            binary=args.decoder_binary
        )

        self.alpha = nn.Parameter(torch.rand(1))

    def attention_pooling(self, v, q, att_map):
        att_map = att_map.squeeze(-1)
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        return fusion_logits

    def forward(self, v_d_gcn, v_d_atomic_embedding, v_p_cnn, v_p_esm, mode="train"):
        # Protein extractor
        if self.is_is_ProteinCNN:
            v_p = self.protein_extractor(v_p_cnn)
        elif self.is_esm_Protein:
            v_p = self.protein_extractor(v_p_esm)

        # LigandGCN case
        if self.is_LigandGCN:
            v_d = self.drug_extractor(v_d_gcn)
            v_p_c = v_p
            v1, q1, att1 = self.gcn_bcn(v_d, v_p_c)

            weighted_att_maps1 = att1 * self.alpha
            fusion_logits1 = self.attention_pooling(v1, q1, weighted_att_maps1)

            f = fusion_logits1

        # Unimol ligand case
        if self.is_unimol_Ligand:
            v_p_u = v_p
            v2, q2, att2 = self.bcn(v_d_atomic_embedding, v_p_u)
            weighted_att_maps2 = att2 * (1 - self.alpha)
            fusion_logits2 = self.attention_pooling(v2, q2, weighted_att_maps2)

            f = fusion_logits2 if not self.is_LigandGCN else torch.cat((f, fusion_logits2), dim=-1)

        # Final score calculation using MLP
        score = self.mlp_classifier(f)

        if mode == "train":
            return f, score
        elif mode == "eval":
            return v_d, v_p, score, att1, att2


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=512, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        original_mask = (node_feats[:, :-1].sum(dim=1) != 0).float().unsqueeze(-1)

        node_feats = self.init_transform(node_feats)
        node_feats = node_feats * original_mask
        node_feats = self.gnn(batch_graph, node_feats)
        node_feats = node_feats * original_mask

        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True, is_ProteinCNN=False):
        super(ProteinCNN, self).__init__()
        self.is_ProteinCNN = is_ProteinCNN
        
        if is_ProteinCNN:
            if padding:
                self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
            else:
                self.embedding = nn.Embedding(26, embedding_dim)

            in_ch = [embedding_dim] + num_filters
            self.in_ch = in_ch[-1]
            kernels = kernel_size
            self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
            self.bn1 = nn.BatchNorm1d(in_ch[1])
            self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
            self.bn2 = nn.BatchNorm1d(in_ch[2])
            self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
            self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        if self.is_ProteinCNN:
            v = self.embedding(v.long())
            v = v.transpose(2, 1)
            v = self.bn1(F.relu(self.conv1(v)))
            v = self.bn2(F.relu(self.conv2(v)))
            v = self.bn3(F.relu(self.conv3(v)))
            v = v.view(v.size(0), v.size(2), -1)
            return v
        else:
            return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.fc3 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

