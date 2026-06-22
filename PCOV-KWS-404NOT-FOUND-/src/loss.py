import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────
# USAGE GUIDE
# SV Head  → SphereFace2Loss(sv_embedding_dim, num_speakers)
# # KWS Head → KWSLoss() using CrossEntropyLoss
# AAMSoftmax provided as alternative for SV if needed
# ─────────────────────────────────────────────


class SphereFace2Loss(nn.Module):
    """
    SphereFace2-based Binary Cosine Classification Loss.
    Pan & Huang (PCOV-KWS, 2026) — key training criterion.
    Transforms multi-class into multiple binary classifications.
    Eliminates inter-category competition between speaker classes.

    Used for: SV Head training
    Input: L2-normalized 192-dim speaker embeddings
    num_classes: number of speaker identities in training set
    """
    def __init__(self, embedding_dim, num_classes,
                 margin=0.5, scale=32.0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        # Defensive L2 normalize — SVHead already does this
        # but enforced here for training safety
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Normalize class weight vectors onto hypersphere
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity scores: (Batch, num_classes)
        cosine = F.linear(embeddings, weight_norm)

        # One-hot target matrix
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply angular margin to target class only
        cosine_m = cosine - self.margin * one_hot
        output = self.scale * cosine_m

        # Binary cross-entropy per class — SphereFace2 key trait
        return F.binary_cross_entropy_with_logits(output, one_hot)


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax — ArcFace.
    Deng et al., CVPR 2019. Industry standard for SV.
    Alternative to SphereFace2 for SV head if needed.

    Used for: SV Head (alternative to SphereFace2)
    Input: L2-normalized embeddings
    """
    def __init__(self, embedding_dim, num_classes,
                 margin=0.2, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Defensive normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return F.cross_entropy(output, labels)


class KWSLoss(nn.Module):
    """
    Keyword Spotting Loss — Standard Cross Entropy.
    Mutually exclusive 4-class classification:
    [0: Target Word, 1: Silence, 2: Unknown, 3: Hard-Negative]
    
    NOTE: Each audio clip must have exactly ONE label.
    Do NOT use this if clips can have multiple simultaneous labels.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # logits: (Batch, 4)
        # labels: (Batch,) — integer class indices
        assert logits.shape[1] == self.num_classes, \
            f"Expected {self.num_classes} classes, got {logits.shape[1]}"
        return self.criterion(logits, labels.long())



class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss (Hinton et al., 2015).
    Forces the tiny Student model to mimic the soft probabilities of a massive Teacher model.
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        assert temperature > 0, "Temperature must be strictly positive to prevent NaN/Inverted gradients."
        assert 0 <= alpha <= 1, "Alpha must be bounded between 0.0 and 1.0."
        
        self.T = temperature  
        self.alpha = alpha    
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    
    def forward(self, student_logits, teacher_logits, labels):
        # [THE FIX] Fail-fast shape assertion
        assert student_logits.shape == teacher_logits.shape, \
            f"KWS KD Error: Student shape {student_logits.shape} must match Teacher shape {teacher_logits.shape}"
            
        # 1. Hard Loss
        hard_loss = self.criterion_ce(student_logits, labels)
        
        # 2. Soft Loss (KL Divergence)
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1).detach()
        
        soft_loss = self.criterion_kl(student_log_probs, teacher_probs) * (self.T ** 2)
        
        # 3. Blend them together
        return (1. - self.alpha) * hard_loss + self.alpha * soft_loss

class SVEmbeddingKDLoss(nn.Module):
    """
    Knowledge Distillation for Speaker Verification (Embeddings).
    Forces the Student's 192D embedding vector to align exactly with 
    the heavy Teacher's (ECAPA-TDNN) embedding vector.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        assert 0 <= alpha <= 1, "Alpha must be bounded between 0.0 and 1.0."
        self.alpha = alpha
        self.criterion_cos = nn.CosineEmbeddingLoss()


    def forward(self, student_emb, teacher_emb, hard_sv_loss):
        # [THE FIX] Fail-fast shape assertion
        assert student_emb.shape == teacher_emb.shape, \
            f"SV KD Error: Student shape {student_emb.shape} must match Teacher shape {teacher_emb.shape}"
            
        # Student embedding retains gradient flow
        student_emb = F.normalize(student_emb, p=2, dim=1)
        
        # Teacher embedding graph is severed.
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1).detach()
        
        # Target tensor of 1s means "make these two vectors identical"
        target = torch.ones(student_emb.size(0)).to(student_emb.device)
        
        soft_loss = self.criterion_cos(student_emb, teacher_emb, target)
        
        return (1. - self.alpha) * hard_sv_loss + self.alpha * soft_loss
