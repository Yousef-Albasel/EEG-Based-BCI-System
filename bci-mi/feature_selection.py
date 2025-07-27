
import os
import numpy as np
from scipy.stats import ttest_1samp
import torch

def compute_relation_spectrum(W1, WG1):
    W1 = W1.T  # we need shape (16,32) for indexing k,i
    # W1[k,i] means row=k, col=i
    WG1 = WG1  # shape (16,16)

    alphas = []
    for i in range(32):
        s = 0.0
        for k in range(16):
            s += WG1[k,k].item() * (W1[k,i].item() ** 2)
        alphas.append(s)

    betas = []
    for i in range(32):
        for j in range(i+1, 32):
            s = 0.0
            for k in range(16):
                for l in range(16):
                    if l != k:
                        s += WG1[k,l].item() * W1[k,i].item() * W1[l,j].item()
            betas.append(s)

    return np.array(alphas + betas, dtype=np.float32)


def index_to_term(idx):
    # 0-31 are squared terms
    if idx < 32:
        return ('sq', idx, None)
    else:
        # compute pair indices for β
        pair_idx = idx - 32
        i = 0
        count = 0
        while True:
            for j in range(i+1, 32):
                if count == pair_idx:
                    return ('int', i, j)
                count += 1
            i += 1


class FeatureTransformLayer:
    def __init__(self, selected_indices):
        self.selected_terms = [index_to_term(i) for i in selected_indices]

    def transform(self, x_batch):
        # x_batch: (batch, 32)
        features = []
        for term in self.selected_terms:
            ttype, i, j = term
            if ttype == 'sq':
                features.append(x_batch[:, i] ** 2)
            else:  # interaction
                features.append(x_batch[:, i] * x_batch[:, j])
        return torch.stack(features, dim=1)  # (batch, m)

def collect_subject_spectra(subjects_indices, ddfilter_dir, compute_relation_spectrum_fn):
    all_spectra = []
    for subj in subjects_indices:
        state_dict = torch.load(
            os.path.join(ddfilter_dir, f"ddfilter_{subj}_v2.pth"),
            map_location='cpu'
        )
        W1 = state_dict['reduce.weight'].T        # (32,16)
        WG1 = state_dict['dd.weight'].T           # (16,16)
        spectrum = compute_relation_spectrum_fn(W1, WG1)
        all_spectra.append(spectrum)
    all_spectra = np.stack(all_spectra, axis=0)
    print("Collected spectra shape:", all_spectra.shape)
    return all_spectra

def select_common_features(all_spectra, alpha_threshold=0.1, save_path=None):
    p_values = []
    selected_indices = []
    for term_idx in range(all_spectra.shape[1]):
        coeffs = all_spectra[:, term_idx]
        t_stat, p_val = ttest_1samp(coeffs, popmean=0.0)
        p_values.append(p_val)
        if p_val < alpha_threshold:
            selected_indices.append(term_idx)
    p_values = np.array(p_values)
    selected_indices = np.array(selected_indices)
    print(f"Selected {len(selected_indices)} common features out of {all_spectra.shape[1]}")
    if save_path is not None:
        np.save(save_path, selected_indices)
        print(f"Saved selected indices to {save_path}")
    return selected_indices, p_values