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
nome_extra = f"_A_k10_{TOLERANCE}-A2BA2"
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
        self.essays = load_dataset("igorcs/C4-A", trust_remote_code=True)['test']
    elif split in ["test-grade-subb", "test-sub-subb"]:
        self.essays = load_dataset("igorcs/C4-B", trust_remote_code=True)['test']
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
        label = (int(self.essays[idx]['cohesive']), int(self.essays[idx]['repetitions']), int(self.essays[idx]['inadequacies']), int(self.essays[idx]['monoblock']) )
    else:
        if isinstance(self.essays[idx]['grades'], str):
            label = eval(self.essays[idx]['grades'])[3]//40
        else:
            label = self.essays[idx]['grades'][3]//40
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
    self.cohesive = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Cohesive2-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=6,
            )
    self.repetitions = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Repetitions-B",#"igorcs/Repetitions-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=5,
            )
    self.inadequacies = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Inadequacies2-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=5,
            )
    self.monoblock = AutoModelForSequenceClassification.from_pretrained(
                "igorcs/Monoblock-A",
                cache_dir="/tmp/aes_enem2",
                num_labels=2,
            )

  def forward(self, x):
    x = list(x)
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    output1 = self.cohesive(input_ids=x[0], token_type_ids=x[1], 
                        attention_mask=x[2])
    output2 = self.repetitions(input_ids=x[0], token_type_ids=x[1], 
                        attention_mask=x[2])
    output3 = self.inadequacies(input_ids=x[0], token_type_ids=x[1], 
                        attention_mask=x[2])
    output4 = self.monoblock(input_ids=x[0], token_type_ids=x[1], 
                        attention_mask=x[2])
    return (F.softmax(output1.logits, dim=1), F.softmax(output2.logits, dim=1), F.softmax(output3.logits, dim=1), F.softmax(output4.logits,dim=1) )


class MNISTSum2Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.resps_cohesive = []
    self.resps_repetitions = []
    self.resps_inadequacies = []
    self.resps_monoblock = []

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("cohesive", int, input_mapping=list(range(6)))
    self.scl_ctx.add_relation("repetitions", int, input_mapping=list(range(5)))
    self.scl_ctx.add_relation("inadequacies", int, input_mapping=list(range(5)))
    self.scl_ctx.add_relation("monoblock", int, input_mapping=list(range(2)))
    #self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("nota(0) :- cohesive(0)")
    self.scl_ctx.add_rule("nota(1) :- cohesive(1), repetitions(b), inadequacies(c), b>=0, c>=0")
    self.scl_ctx.add_rule("nota(1) :- cohesive(a), repetitions(0), inadequacies(c), a>=1, c>=0")
    self.scl_ctx.add_rule("nota(1) :- cohesive(a), repetitions(b), inadequacies(0), a>=1, b>=0")

    self.scl_ctx.add_rule("nota(2) :- cohesive(2), repetitions(b), inadequacies(c), b>=1, c>=1")
    self.scl_ctx.add_rule("nota(2) :- cohesive(a), repetitions(1), inadequacies(c), a>=2, c>=1")
    self.scl_ctx.add_rule("nota(2) :- cohesive(a), repetitions(b), inadequacies(1), a>=2, b>=1")
    self.scl_ctx.add_rule("nota(2) :- cohesive(a), repetitions(b), inadequacies(c), monoblock(0), a>=3, b>=2, c>=2")

    self.scl_ctx.add_rule("nota(3) :- cohesive(3), repetitions(b), inadequacies(c), b>=2, c>=2")
    self.scl_ctx.add_rule("nota(3) :- cohesive(a), repetitions(2), inadequacies(c), a>=3, c>=2")
    self.scl_ctx.add_rule("nota(3) :- cohesive(a), repetitions(b), inadequacies(2), a>=3, b>=2")

    self.scl_ctx.add_rule("nota(4) :- cohesive(4), repetitions(b), inadequacies(c), b>=3, c>=3")
    self.scl_ctx.add_rule("nota(4) :- cohesive(a), repetitions(3), inadequacies(c), a>=4, c>=3")
    self.scl_ctx.add_rule("nota(4) :- cohesive(a), repetitions(b), inadequacies(3), a>=4, b>=3")
    

    self.scl_ctx.add_rule("nota(5) :- cohesive(5), repetitions(4), inadequacies(4)")
    # The `sum_2` logical reasoning module
    self.sum_2 = self.scl_ctx.forward_function("nota", output_mapping=[(i,) for i in range(6)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    texto = x
    # First recognize the two digits
    resposta_a, resposta_b, resposta_c, resposta_d = self.mnist_net(texto) # Tensor
    self.resps_cohesive.extend(resposta_a)
    self.resps_repetitions.extend(resposta_b)
    self.resps_inadequacies.extend(resposta_c)
    self.resps_monoblock.extend(resposta_d)
    #b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    # Then execute the reasoning module; the result is a size 19 tensor
    return self.sum_2(cohesive=resposta_a, repetitions=resposta_b, inadequacies=resposta_c, monoblock=resposta_d)#, digit_2=b_distrs) # Tensor 64 x 19

  def reset_memory(self):
      self.resps_cohesive = []
      self.resps_repetitions = []
      self.resps_inadequacies = []
      self.resps_monoblock = []


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
          torch.save(self.network.mnist_net.state_dict(), 'RedeTreinada_C4'+nome_extra+'.pth')
          print("Modelo salvo")
      if (stage == 'validation') and (self.melhor_iteracao != epoch) and (QWK <= self.melhor_QWK_valid):
          self.tolerance -= 1
      if (stage.startswith('test')) and (self.melhor_iteracao == epoch):
          self.dic[f'y_{num_test}'] = [t.item() for t in y]
          self.dic[f'y_hat_{num_test}'] = [t.item() for t in y_hat]
    self.dic[f"loss_{stage}"] = test_loss
    self.dic[f"{stage}_acc"] = perc.item()
    self.dic[f"{stage}_qwk"] = QWK

  def testar_listas(self, l1, l2):
    elemento1 = l1[0]
    l1_igual = True
    for e in l1:
        if e != elemento1:
            l1_igual = False
            break
    elemento2 = l2[0]
    l2_igual = True
    for e in l2:
        if e != elemento2:
            l2_igual = False
            break
    if l1_igual and l2_igual and elemento1 == elemento2:
        return False
    else:
        return True
  def test_sub_network(self, dataset_using, epoch, stage, num_test):
    num_items = len(dataset_using.dataset)
    test_loss = 0
    correct = 0
    y_0, y_1, y_2, y_3 = [], [], [], []
    y_hat_0, y_hat_1, y_hat_2, y_hat_3 = [], [], [], []
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
        for idx, vetor3 in enumerate(output[2]):
            pred2 = vetor3.max(0, keepdim=True)[1]
            y_hat_2.append(pred2.item())
        for idx, vetor4 in enumerate(output[3]):
            pred3 = vetor4.max(0, keepdim=True)[1]
            y_hat_3.append(pred3.item())
        y_0.extend([a.item() for a,_,_,_ in target])
        y_1.extend([a.item() for _,a,_,_ in target])
        y_2.extend([a.item() for _,_,a,_ in target])
        y_3.extend([a.item() for _,_,_,a in target])
        #print("Y_0:", y_0)
        #print("Y_hat_0:", y_hat_0)
        #print("Y_1:", y_1)
        #print("Y_hat_1:", y_hat_1)
        #print("Y_2:", y_2)
        #print("Y_hat_2:", y_hat_2)
        #print("Y_3:", y_3)
        #print("Y_hat_3:", y_hat_3)
        perc_0 = accuracy_score(y_0, y_hat_0)*100
        #print("Computei o ACC0")
        perc_1 = accuracy_score(y_1, y_hat_1)*100
        #print("Computei o ACC1")
        perc_2 = accuracy_score(y_2, y_hat_2)*100
        #print("Computei o ACC2")
        perc_3 = accuracy_score(y_3, y_hat_3)*100
        #print("Computei o ACC3")
        QWK_0 = cohen_kappa_score(y_0, y_hat_0, weights='quadratic', labels=[0,1,2,3,4,5])
        #print("foi o 0")
        QWK_1 = cohen_kappa_score(y_1, y_hat_1, weights='quadratic', labels=[0,1,2,3,4])
        #print("foi o 1")
        QWK_2 = cohen_kappa_score(y_2, y_hat_2, weights='quadratic', labels=[0,1,2,3,4])
        #print("foi o 2")
        if self.testar_listas(y_3, y_hat_3):
            QWK_3 = cohen_kappa_score(y_3, y_hat_3, weights='quadratic', labels=[0,1])
        else:
            QWK_3 = 1.0
        #print("erro foi no 3")
        iter.set_description(f"[{stage} Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy:({perc_0:.2f}, {perc_1:.2f}, {perc_2:.2f}, {perc_3:.2f})  QWK: {QWK_0:.2f}, {QWK_1:.2f}, {QWK_2}, {QWK_3}")
      if (stage.startswith('test')) and (self.melhor_iteracao == epoch):
          self.dic[f'y_{num_test}_cohesive'] = y_0
          self.dic[f'y_{num_test}_repetitions'] = y_1
          self.dic[f'y_{num_test}_inadequacies'] = y_2
          self.dic[f'y_{num_test}_monoblock'] = y_3
          self.dic[f'y_hat_{num_test}_cohesive'] = y_hat_0
          self.dic[f'y_hat_{num_test}_repetitions'] = y_hat_1
          self.dic[f'y_hat_{num_test}_inadequacies'] = y_hat_2
          self.dic[f'y_hat_{num_test}_monoblock'] = y_hat_3
    self.dic[f"loss_{stage}"] = test_loss
    self.dic[f"{stage}_acc_cohesive"] = perc_0
    self.dic[f"{stage}_acc_repetitions"] = perc_1
    self.dic[f"{stage}_acc_inadequacies"] = perc_2
    self.dic[f"{stage}_acc_monoblock"] = perc_3
    self.dic[f"{stage}_qwk_cohesive"] = QWK_0
    self.dic[f"{stage}_qwk_repetitions"] = QWK_1
    self.dic[f"{stage}_qwk_inadequacies"] = QWK_2
    self.dic[f"{stage}_qwk_monoblock"] = QWK_3


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
    nome_arquivo = 'performances_loaded_dois_berts_c4'+nome_extra+'.csv'
    with open(nome_arquivo, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(self.lista_performances)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--n-epochs", type=int, default=2)
  parser.add_argument("--batch-size-train", type=int, default=1)
  parser.add_argument("--batch-size-test", type=int, default=1)
  parser.add_argument("--learning-rate", type=float, default=0.000001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="diffminmaxprob")
  parser.add_argument("--top-k", type=int, default=3)
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


