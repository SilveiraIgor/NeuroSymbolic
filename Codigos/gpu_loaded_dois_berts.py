import os
import random
from typing import *
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

import scallopy

TOKENIZER_NAME = f"neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
TOLERANCE = 20
nome_extra = "_A_longo_k10"
device = "cuda" if torch.accelerator.is_available() else "cpu"
class MNISTSum2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    split: str,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.split_name = split
    if split == "train":
        self.essays = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="/tmp/aes_enem", trust_remote_code=True)['train']
    elif split == "test":
        self.essays = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="/tmp/aes_enem", trust_remote_code=True)['test']
    elif split in ["test-grade-suba", "test-sub-suba"]:
        self.essays = load_dataset("igorcs/C1-A", trust_remote_code=True)['test']
    elif split in ["test-grade-subb", "test-sub-subb"]:
        self.essays = load_dataset("igorcs/C1-B", trust_remote_code=True)['test']
    else:
        self.essays =  self.essays = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="/tmp/aes_enem", trust_remote_code=True)['validation']


  def __len__(self):
     return len(self.essays)

  def __getitem__(self, idx):
    # Get two data points
    #(a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    #(b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]
    tokenized_text = tokenizer(
                self.essays[idx]["essay_text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
    if self.split_name.startswith("test-sub-"):
        label = (int(self.essays[idx]['syntax']), int(self.essays[idx]['mistakes']) )
    else:
        if isinstance(self.essays[idx]['grades'], str):
            label = eval(self.essays[idx]['grades'])[0]//40
        else:
            label = self.essays[idx]['grades'][0]//40
    # Each data has two images and the GT is the sum of two digits
    return (tokenized_text, label)#(a_img, b_img, a_digit + b_digit)

  @staticmethod
  def collate_fn(batch):
    input_ids = torch.stack([item[0]['input_ids'][0] for item in batch])
    token_type = torch.stack([item[0]['token_type_ids'][0] for item in batch])
    attention_mask = torch.stack([item[0]['attention_mask'][0] for item in batch])
    digits = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return ((input_ids, token_type, attention_mask), digits)


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="train",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=False
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )

  validation_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="validation",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )

  sub_a_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test-grade-suba",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )
  sub_b_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test-grade-subb",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )
  sub_sub_a_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test-sub-suba",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )
  sub_sub_b_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test-sub-subb",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )

  return train_loader, validation_loader, [test_loader, sub_a_loader, sub_b_loader, sub_sub_a_loader, sub_sub_b_loader]


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.sintaxe = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Syntax-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=5,
            )
    self.desvios = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Mistakes-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=4,
            )

  def forward(self, x):
    output1 = self.sintaxe(input_ids=x[0].to(device), token_type_ids=x[1].to(device), 
                        attention_mask=x[2].to(device))
    output2 = self.desvios(input_ids=x[0].to(device), token_type_ids=x[1].to(device), 
                        attention_mask=x[2].to(device))

    return (F.softmax(output1.logits, dim=1), F.softmax(output2.logits, dim=1))


class MNISTSum2Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.resps_A = []
    self.resps_B = []

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(5)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(4)))
    #self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_2(0) :- digit_1(0)")
    self.scl_ctx.add_rule("sum_2(1) :- digit_1(1), digit_2(0)")
    self.scl_ctx.add_rule("sum_2(2) :- digit_1(1), digit_2(b), b>=1")
    self.scl_ctx.add_rule("sum_2(2) :- digit_1(a), digit_2(0), a>=2")
    self.scl_ctx.add_rule("sum_2(3) :- digit_1(2), digit_2(b), b>=1")
    self.scl_ctx.add_rule("sum_2(3) :- digit_1(a), digit_2(1), a>=2")
    self.scl_ctx.add_rule("sum_2(4) :- digit_1(3), digit_2(b), b>=2")
    self.scl_ctx.add_rule("sum_2(4) :- digit_1(a), digit_2(2), a>=3")
    self.scl_ctx.add_rule("sum_2(5) :- digit_1(4), digit_2(3)")
    # The `sum_2` logical reasoning module
    self.sum_2 = self.scl_ctx.forward_function("sum_2", output_mapping=[(i,) for i in range(6)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    texto = x
    # First recognize the two digits
    resposta_a, resposta_b = self.mnist_net(texto) # Tensor 64 x 10
    self.resps_A.extend(resposta_a)
    self.resps_B.extend(resposta_b)
    #b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    # Then execute the reasoning module; the result is a size 19 tensor
    return self.sum_2(digit_1=resposta_a, digit_2=resposta_b)#, digit_2=b_distrs) # Tensor 64 x 19

  def reset_memory(self):
      self.resps_A = []
      self.resps_B = []


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, validation_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSum2Net(provenance, k).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.validation_loader = validation_loader
    self.test_loader = test_loader
    self.melhor_QWK_valid = -2
    self.melhor_iteracao = 0
    self.tolerance = TOLERANCE
    self.dic = {}
    self.lista_performances = []
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.dic['epoca'] = epoch
    self.network.train()
    self.network.reset_memory()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      #target = target.to(device)
      self.optimizer.zero_grad()
      output = self.network(data).cpu()
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    self.dic['loss_train'] = loss.item()
    self.dic['concodancia_train'] = self.medir_concordancia()


  def medir_concordancia(self):
    resp_A = self.network.resps_A#[0].max(dim=1, keepdim=False)[1]
    resp_B = self.network.resps_B#[0].max(dim=1, keepdim=False)[1]
    args_A = torch.tensor([t.argmax() for t in resp_A])
    args_B = torch.tensor([t.argmax() for t in resp_B])
    iguais = (args_A == args_B).sum().item()
    print( f"Total de concordancias: {iguais}/{len(args_A)} ({iguais/len(args_A):.2f})" )
    return iguais/len(args_A)

  def test_whole_network(self, dataset_using, epoch, stage, num_test):
    num_items = len(dataset_using.dataset)
    test_loss = 0
    correct = 0
    y = []
    y_hat = []
    with torch.no_grad():
      iter = tqdm(dataset_using, total=len(dataset_using))
      for (data, target) in iter:
        #data, target = data.to(device), target#.to(device)
        output = self.network(data).cpu()
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        y.extend(torch.flatten(target).numpy())
        y_hat.extend(torch.flatten(pred).numpy())
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        QWK = cohen_kappa_score(y, y_hat, weights='quadratic', labels=[0,1,2,3,4,5])
        iter.set_description(f"[{stage} Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%) QWK: {QWK:.2f}")
      if (stage == "validation") and (QWK > self.melhor_QWK_valid):
          self.melhor_QWK_valid = QWK
          self.melhor_iteracao = epoch
          self.tolerance = TOLERANCE
          print("Vou salvar esse modelo")
          torch.save(self.network.mnist_net.state_dict(), 'RedeTreinada'+nome_extra+'.pth')
          print("Modelo salvo")
      if (stage == 'validation') and (QWK < self.melhor_QWK_valid):
          self.tolerance -= 1
      if (stage.startswith('test')) and (self.melhor_iteracao == epoch):
          self.dic[f'y_{num_test}'] = [t.item() for t in y]
          self.dic[f'y_hat_{num_test}'] = [t.item() for t in y_hat]
    self.dic[f"loss_{stage}"] = test_loss
    self.dic[f"{stage}_acc"] = perc.item()
    self.dic[f"{stage}_qwk"] = QWK

  def test_sub_network(self, dataset_using, epoch, stage, num_test):
    num_items = len(dataset_using.dataset)
    test_loss = 0
    correct = 0
    y_0, y_1 = [], []
    y_hat_0, y_hat_1 = [], []
    with torch.no_grad():
      iter = tqdm(dataset_using, total=len(dataset_using))
      for (data, target) in iter:
        output = self.network.mnist_net(data)
        test_loss += 0
        for idx, vetor1 in enumerate(output[0]):
            pred0 = vetor1.max(0, keepdim=True)[1]
            y_hat_0.append(pred0.item())
        for idx, vetor2 in enumerate(output[1]):
            pred1 = vetor2.max(0, keepdim=True)[1]
            y_hat_1.append(pred1.item())
        y_0.extend([a.item() for a,_ in target])
        y_1.extend([a.item() for _,a in target])
        perc_0 = accuracy_score(y_0, y_hat_0)*100
        perc_1 = accuracy_score(y_1, y_hat_1)*100
        QWK_0 = cohen_kappa_score(y_0, y_hat_0, weights='quadratic', labels=[0,1,2,3,4,5])
        QWK_1 = cohen_kappa_score(y_1, y_hat_1, weights='quadratic', labels=[0,1,2,3,4,5])
        iter.set_description(f"[{stage} Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy:({perc_0:.2f}, {perc_1:.2f})  QWK: {QWK_0:.2f}, {QWK_1:.2f}")
      if (stage.startswith('test')) and (self.melhor_iteracao == epoch):
          self.dic[f'y_{num_test}_syntax'] = y_0
          self.dic[f'y_{num_test}_mistake'] = y_1
          self.dic[f'y_hat_{num_test}_syntax'] = y_hat_0
          self.dic[f'y_hat_{num_test}_mistake'] = y_hat_1
    self.dic[f"loss_{stage}"] = test_loss
    self.dic[f"{stage}_acc_syntax"] = perc_0
    self.dic[f"{stage}_acc_mistakes"] = perc_1
    self.dic[f"{stage}_qwk_syntax"] = QWK_0
    self.dic[f"{stage}_qwk_mistakes"] = QWK_1


  def test(self, epoch, stage):
    self.dic['epoca'] = epoch
    self.network.eval()
    self.network.reset_memory()
    if stage.startswith("test"):
        num_test = int(stage[5:])
        dataset_using = self.test_loader[num_test]
        if num_test < 3:
            self.test_whole_network(dataset_using, epoch, stage, num_test)
        else:
            self.test_sub_network(dataset_using, epoch, stage, num_test)
    else:
        num_test = ""
        dataset_using = self.validation_loader
        self.test_whole_network(dataset_using, epoch, stage, num_test)


  def salvar_performance(self):
      self.lista_performances.append(self.dic)
      self.dic = {}

  def train(self, n_epochs):
    self.test(0, "test-0")
    self.test(0, "test-1")
    self.test(0, "test-2")
    self.test(0, "test-3")
    self.test(0, "test-4")
    self.salvar_performance()
    epoch = 1
    #self.tolerance = 0
    while (self.tolerance > 0):
      self.train_epoch(epoch)
      self.test(epoch, "validation")
      for i in range(len(self.test_loader)):
        self.test(epoch, f"test-{i}")
      epoch += 1
      self.salvar_performance()
    keys = self.lista_performances[1].keys()
    nome_arquivo = 'performances_loaded_dois_berts'+nome_extra+'.csv'
    with open(nome_arquivo, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(self.lista_performances)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--n-epochs", type=int, default=2)
  parser.add_argument("--batch-size-train", type=int, default=8)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.000001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=10)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, validation_loader, test_loaders = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test)
  # Create trainer and train
  trainer = Trainer(train_loader, validation_loader, test_loaders , learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)


