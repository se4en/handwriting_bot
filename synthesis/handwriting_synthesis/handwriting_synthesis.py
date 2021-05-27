import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import math
import torch.nn as nn
from torch.distributions import bernoulli, uniform
from utils import plot_stroke


def stable_softmax(X, dim=2):
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat

def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
        y_hat.device
    )
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample

class HandWritingSynthesisNet(nn.Module):

    def __init__(self, hidden_size=400, n_layers=3, output_size=121, window_size=77):
        super(HandWritingSynthesisNet, self).__init__()
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        K = 10
        self.EOS = False
        self._phi = []

        self.lstm_1 = nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )
        self.lstm_3 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )

        self.window_layer = nn.Linear(hidden_size, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa

    def one_hot_encoding(self, text):
        N = text.shape[0]
        U = text.shape[1]
        encoding = text.new_zeros((N, U, self.vocab_size))
        for i in range(N):
            encoding[i, torch.arange(U), text[i].long()] = 1.0
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask, is_map):
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(text.shape[1], dtype=torch.float32, device=text.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True
        phi = (phi * text_mask).unsqueeze(2)
        if is_map:
            self._phi.append(phi.squeeze(dim=2).unsqueeze(1))

        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def init_weight(self):
        k = math.sqrt(1.0 / self.hidden_size)
        for param in self.lstm_1.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_2.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        for param in self.lstm_3.parameters():
            nn.init.uniform_(param, a=-k, b=k)

        nn.init.uniform_(self.window_layer.weight, a=-0.01, b=0.01)
        nn.init.constant_(self.window_layer.bias, 0.0)

        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(
        self,
        inputs,
        text,
        text_mask,
        initial_hidden,
        prev_window_vec,
        prev_kappa,
        is_map=False,
    ):

        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t: t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2),
                prev_kappa,
                text,
                text_mask,
                is_map,
            )

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa

    def generate(
        self,
        inp,
        text,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vector,
        kappa,
        bias,
        is_map=False,
        prime=False,
    ):
        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = inp.shape[0]
            if prime:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, prime_text, prime_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                self.EOS = False
                inp = inp.new_zeros(batch_size, 1, 3)
                _, window_vector, kappa = self.init_hidden(batch_size, inp.device)

            while not self.EOS and seq_len < 2000:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, text, text_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                y_hat = y_hat.squeeze()
                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        return gen_seq

def generate_sequence(
    model_path,
    char_seq,
    device,
    char_to_id,
    idx_to_char,
    bias,
    prime,
    prime_seq,
    real_text,
    is_map,
    batch_size=1,
):
    model = HandWritingSynthesisNet(window_size=len(char_to_id))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    if prime:
        inp = prime_seq
        real_seq = np.array(list(real_text))
        idx_arr = [char_to_id[char] for char in real_seq]
        prime_text = np.array([idx_arr for i in range(batch_size)]).astype(np.float32)
        prime_text = torch.from_numpy(prime_text).to(device)
        prime_mask = torch.ones(prime_text.shape).to(device)
    else:
        prime_text = None
        prime_mask = None
        inp = torch.zeros(batch_size, 1, 3).to(device)

    char_seq = np.array(list(char_seq + "  "))
    text = np.array(
        [[char_to_id[char] for char in char_seq] for i in range(batch_size)]
    ).astype(np.float32)
    text = torch.from_numpy(text).to(device)

    text_mask = torch.ones(text.shape).to(device)

    hidden, window_vector, kappa = model.init_hidden(batch_size, device)
    gen_seq = model.generate(
        inp,
        text,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vector,
        kappa,
        bias,
        is_map,
        prime=prime,
    )

    length = len(text_mask.nonzero())

    if is_map:
        phi = torch.cat(model._phi, dim=1).cpu().numpy()
        phi = phi[0].T
    else:
        phi = []

    return gen_seq, phi

def train_offset_normalization(data):

    mean = data[:, :, 1:].mean(axis=(0, 1))
    data[:, :, 1:] -= mean
    std = data[:, :, 1:].std(axis=(0, 1))
    data[:, :, 1:] /= std

    return mean, std, data


def valid_offset_normalization(mean, std, data):

    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data


def data_denormalization(mean, std, data):

    data[:, :, 1:] *= std
    data[:, :, 1:] += mean

    return data


def data_normalization(data):

    mean = data[:, 1:].mean(axis=0)
    data[:, 1:] -= mean
    std = data[:, 1:].std(axis=0)
    data[:, 1:] /= std

    return mean, std, data

class HandwritingDataset(Dataset):

    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        strokes = np.load(data_path + 'strokes.npy', allow_pickle=True, encoding='bytes')
        with open(data_path + 'sentences.txt') as file:
            texts = file.read().splitlines()

        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths)
        n_total = len(strokes)

        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        char_seqs = [list(char_seq) for char_seq in texts]
        char_seqs = np.asarray(char_seqs)

        char_lens = [len(char_seq) for char_seq in char_seqs]
        max_char_len = np.max(char_lens)

        mask_shape = (n_total, max_char_len) 
        char_mask = np.zeros(mask_shape, dtype=np.float32)

        inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
        inp_text[:, :] = ' '

        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, (seq_len, text_len) in enumerate(zip(lengths, char_lens)):
            mask[i, :seq_len] = 1.
            data[i, :seq_len] = strokes[i]
            char_mask[i, :text_len] = 1.
            inp_text[i, :text_len] = char_seqs[i]

        self.id_to_char, self.char_to_id = self.build_vocab(inp_text)
        self.vocab_size = len(self.id_to_char)

        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        inp_text = inp_text[idx_permute]
        char_mask = char_mask[idx_permute]

        if debug:
            data = data[:64]
            mask = mask[:64]
            inp_text = inp_text[:64]
            char_mask = char_mask[:64]

        n_train = int(0.9 * data.shape[0])
        self._data = data
        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.texts = inp_text[:n_train]
            self.char_mask = char_mask[:n_train]
            0, 1, self.dataset = train_offset_normalization(
                self.dataset)

        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.texts = inp_text[n_train:]
            self.char_mask = char_mask[n_train:]
            self.dataset = valid_offset_normalization(
                0, 1, self.dataset)

    def __len__(self):
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        return np.array([self.id_to_char[id] for id in id_seq])

    def char_to_idx(self, char_seq):
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.float32)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text)
        unique_char = sorted(counter)
        vocab_size = len(unique_char)

        id_to_char = dict(zip(np.arange(vocab_size), unique_char))
        char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
        return id_to_char, char_to_id

    def __getitem__(self, idx):

        mask = torch.from_numpy(self.mask[idx])

        if self.text_req:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])

            target = torch.from_numpy(self.dataset[idx])
            text = torch.from_numpy(self.char_to_idx(self.texts[idx]))
            char_mask = torch.from_numpy(self.char_mask[idx])
            return (input_seq, target, mask, text, char_mask)
        elif self.data_aug:
            seq_len = len(mask.nonzero())
            start = 0
            end = self.max_seq_len

            if seq_len > self.max_seq_len:
                start = np.random.randint(0, high=seq_len - self.max_seq_len)
                end = start + self.max_seq_len

            stroke = self.dataset[idx, start:end, :]

            input_seq = torch.zeros(stroke.shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(stroke[:-1, :])

            target = torch.from_numpy(stroke)
            mask = mask[start:end]

            return (input_seq, target, mask)
        else:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])
            target = torch.from_numpy(self.dataset[idx])
            return (input_seq, target, mask)