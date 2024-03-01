'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        self.num_sentiment_labels = 5
        self.num_paraphrase_labels = 1
        self.num_similarity_labels = 1
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, self.num_sentiment_labels)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, self.num_paraphrase_labels)
        self.similarity_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, self.num_similarity_labels)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # Pass input IDs and attention masks to BERT model
        outputs = self.bert(input_ids, attention_mask)
        # Get embeddings matrix
        last_hidden_state = outputs['last_hidden_state']
        # Get CLS token representation
        cls_token_rep = last_hidden_state[:, 0, :]
        # Returns CLS token representations for both sentiment classification and similarity detection
        return cls_token_rep



    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        cls_token_rep = self.forward(input_ids, attention_mask)
        sentiment_logits = self.sentiment_classifier(cls_token_rep)
        return sentiment_logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        outputs_1 = self.bert(input_ids_1, attention_mask_1)
        outputs_2 = self.bert(input_ids_2, attention_mask_2)
        cls_token_rep_1 = outputs_1["last_hidden_state"][:, 0, :]
        cls_token_rep_2 = outputs_2["last_hidden_state"][:, 0, :]
        combined_embeddings = torch.cat([cls_token_rep_1, cls_token_rep_2], dim=1)
        return self.paraphrase_classifier(combined_embeddings)



    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        outputs_1 = self.bert(input_ids_1, attention_mask_1)
        outputs_2 = self.bert(input_ids_2, attention_mask_2)
        cls_token_rep_1 = outputs_1["last_hidden_state"][:, 0, :]
        cls_token_rep_2 = outputs_2["last_hidden_state"][:, 0, :]
        combined_embeddings = torch.cat([cls_token_rep_1, cls_token_rep_2], dim=1)
        return self.similarity_classifier(combined_embeddings)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    sst_best_dev_acc = 0
    para_best_dev_acc = 0
    sts_best_dev_acc = 0

    cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        sst_train_loss = 0
        sst_num_batches = 0
        para_train_loss = 0
        para_num_batches = 0
        sts_train_loss = 0
        sts_num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            sst_train_loss += loss.item()
            sst_num_batches += 1

        if (config.option == 'pretrain'):
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_input_ids_1, b_mask_1, b_input_ids_2, b_mask_2, b_labels = (
                    batch['token_ids_1'],
                    batch['attention_mask_1'],
                    batch['token_ids_2'],
                    batch['attention_mask_2'],
                    batch['labels']
                )
                b_ids_1 = b_input_ids_1.to(device)
                b_ids_2 = b_input_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                
                # Linear layer to get logits
                logits = model.predict_paraphrase(b_ids_1,b_mask_1, b_ids_2, b_mask_2)
                
                loss = F.cross_entropy(logits, b_labels.view, reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                para_train_loss += loss.item()
                para_num_batches += 1
        else: 
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_input_ids_1, b_mask_1, b_input_ids_2, b_mask_2, b_labels = (
                    batch['token_ids_1'],
                    batch['attention_mask_1'],
                    batch['token_ids_2'],
                    batch['attention_mask_2'],
                    batch['labels']
                )
                b_ids_1 = b_input_ids_1.to(device)
                b_ids_2 = b_input_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                cls_token_rep_1 = model.forward(b_ids_1, b_mask_1)
                cls_token_rep_2 = model.forward(b_ids_2, b_mask_2)
                loss = cosine_loss_fn(cls_token_rep_1, cls_token_rep_2, b_labels)

                loss.backward()
                optimizer.step()

                para_train_loss += loss.item()
                para_num_batches += 1
            
        
        if (config.option == 'pretrain'):
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_input_ids_1, b_mask_1, b_input_ids_2, b_mask_2, b_labels = (
                    batch['token_ids_1'],
                    batch['attention_mask_1'],
                    batch['token_ids_2'],
                    batch['attention_mask_2'],
                    batch['labels']
                )
                b_ids_1 = b_input_ids_1.to(device)
                b_ids_2 = b_input_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                
                # Linear layer to get logits
                logits = model.predict_similarity(b_ids_1,b_mask_1, b_ids_2, b_mask_2)
          
                loss = F.cross_entropy(logits, b_labels.view, reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                sts_train_loss += loss.item()
                sts_num_batches += 1
        else:
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_input_ids_1, b_mask_1, b_input_ids_2, b_mask_2, b_labels = (
                    batch['token_ids_1'],
                    batch['attention_mask_1'],
                    batch['token_ids_2'],
                    batch['attention_mask_2'],
                    batch['labels']
                )
                b_ids_1 = b_input_ids_1.to(device)
                b_ids_2 = b_input_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                cls_token_rep_1 = model.forward(b_ids_1, b_mask_1)
                cls_token_rep_2 = model.forward(b_ids_2, b_mask_2)
                loss = cosine_loss_fn(cls_token_rep_1, cls_token_rep_2, b_labels)

                loss.backward()
                optimizer.step()

                sts_train_loss += loss.item()
                sts_num_batches += 1

        sst_train_loss = sst_train_loss / (sst_num_batches)
        para_train_loss = para_train_loss / (para_num_batches)
        sts_train_loss = sts_train_loss / (sts_num_batches)

        sst_train_acc, sst_train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        

        """
        return (sentiment_accuracy,sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)
        """
        print("Multitask train accuracies")
        _, _, _, para_train_acc, _, _, sts_train_acc, *_ =  model_eval_multitask(sst_train_dataloader,
                                                                                      para_train_dataloader,
                                                                                      sts_train_dataloader, model,
                                                                                      device)
        print("Multitask dev accuracies")
        _, _, _, para_dev_acc, _, _, sts_dev_acc, *_ = model_eval_multitask(sst_dev_dataloader,
                                                                                     para_dev_dataloader,
                                                                                     sts_dev_dataloader, model,
                                                                                     device)

        if sst_dev_acc > sst_best_dev_acc or para_dev_acc > para_best_dev_acc or sts_dev_acc > sts_best_dev_acc:
            if sst_dev_acc > sst_best_dev_acc:
                sst_best_dev_acc = sst_dev_acc
            if para_dev_acc > para_best_dev_acc:
                para_best_dev_acc = para_dev_acc
            if sts_dev_acc > sts_best_dev_acc:
                sts_best_dev_acc = sts_dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: SST train loss :: {sst_train_loss :.3f}, train acc :: {sst_train_acc :.3f}, dev acc :: {sst_dev_acc :.3f}")
        print(f"Epoch {epoch}: Paraphrase train loss :: {para_train_loss :.3f}, train acc :: {para_train_acc :.3f}, dev acc :: {para_dev_acc :.3f}")
        print(f"Epoch {epoch}: STS train loss :: {sts_train_loss :.3f}, train acc :: {sts_train_acc :.3f}, dev acc :: {sts_dev_acc :.3f}")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
