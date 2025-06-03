import clip

import torch
import torch.nn as nn
import torch.nn.functional as F

from mrl_layer import MRL_Linear_Layer

class MRL_CLIP_VQA(nn.Module):
    def __init__(self, clip_model, nesting_list, relative_importance=None):
        super(MRL_CLIP_VQA, self).__init__()
        
        # Extract components from original CLIP
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # For ViT model, we need to get the pre-projection dimensions
        if hasattr(self.visual, 'proj'):
            self.visual_hidden_dim = self.visual.proj.shape[1]  # Input dim to projection
        elif hasattr(self.visual, 'output_dim'):
            self.visual_hidden_dim = self.visual.output_dim
        else:
            self.visual_hidden_dim = 768  # Common dimension for ViT-B
            print("Warning: Could not determine visual hidden dimension, using default 768")
            
        # MRL projection
        self.mrl_visual_projection = MRL_Linear_Layer(
            nesting_list=nesting_list,
            num_classes=self.text_projection.shape[1],  # Match text dimension
            efficient=True,
            bias=False
        )
        
        self.nesting_list = nesting_list
        self.logit_scale = clip_model.logit_scale
        self.relative_importance = relative_importance or [1.0] * len(nesting_list)
    
    def encode_image(self, images):
        # Get image features (pre-projection)
        image_features = self.visual(images)
        
        # Apply MRL projection to get nested embeddings
        nested_embeds = self.mrl_visual_projection(image_features)
        
        # Normalize each granularity level
        normalized_nested_embeds = []
        for embed in nested_embeds:
            normalized_embed = embed / embed.norm(dim=-1, keepdim=True)
            normalized_nested_embeds.append(normalized_embed)
        
        return normalized_nested_embeds
    
    def encode_text(self, text_tokens):
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # Take features from the eot embedding (last token)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection
        
        # Normalize
        x = x / x.norm(dim=-1, keepdim=True)
        
        return x
    
    def forward(self, images, questions):
        # Tokenize questions
        question_tokens = clip.tokenize(questions, truncate=True).to(images.device)
        
        # Get representations
        image_features = self.encode_image(images)
        text_features = self.encode_text(question_tokens)
        
        # Compute contrastive loss at each granularity
        losses = []
        for i, img_feat in enumerate(image_features):
            # Calculate similarity
            logits = img_feat @ text_features.T * self.logit_scale.exp()
            
            # Get labels (diagonal for positive pairs)
            labels = torch.arange(images.shape[0], device=images.device)
            
            # Calculate symmetrical loss
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            losses.append((loss_i + loss_t) / 2)
        
        # Stack losses
        losses = torch.stack(losses)
        
        # Apply relative importance weights
        weights = torch.tensor(self.relative_importance, device=losses.device)
        losses = losses * weights
            
        return losses.sum(), losses