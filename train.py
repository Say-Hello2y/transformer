import math
import time
import os
import torch
import argparse
from distutils.util import strtobool

from torch import nn, optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from preprocess import *
from model.tansformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="transformer",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")


    args = parser.parse_args()

    # fmt: on
    return args







def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# def save_model(model):
#     time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
#     path = os.path.abspath("saved/" + 'transformer'+ time_now + ".pth")
#     path_dir = os.path.abspath("saved/")
#     if not os.path.isdir(path_dir):
#         os.mkdir(path_dir)
#     torch.save(model.module, path)
#     print(
#         "****************************************************************************************************************************************************"
#     )
#     print("save model to {}".format(path))

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters') 
# The model has 55,205,037 trainable parameters
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip,step,track=False):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):  # 遍历train——iter
        src = batch.src
        trg = batch.trg
        # print(trg)
        # print(src.size())
        # print(trg.size())

        optimizer.zero_grad()
        # the French sentence we input has all words except
        # the last, as it is using each word to predict the next
        output = model(src, trg[:, :-1])  # multi sentence parallel training
        # print(trg[:,:-1])
        # print(output.size())
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # output,shape[-1]: voc length
        # print(output_reshape.size())
        trg = trg[:, 1:].contiguous().view(-1)
        # print(trg.size())

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
        if track:
            writer.add_scalar(
                        "losses/train_loss", loss.item(), i + step*len(iterator) + 1
                    )

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss, track=False):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip,step,track)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # f = open('saved/train_loss.txt', 'w')
        # f.write(str(train_losses))
        # f.close()

        # f = open('result/bleu.txt', 'w')
        # f.write(str(bleus))
        # f.close()

        # f = open('result/test_loss.txt', 'w')
        # f.write(str(test_losses))
        # f.close()
        
        if track:                   
            writer.add_scalar(
                        "losses/valid_loss", train_loss, step
                    )
        
            writer.add_scalar(
                        "losses/bleu", bleu, step
                    )

        

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    args = parse_args()
    if args.track:
        import wandb  # visualize tool
        print('use wandb')
        time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        run_name = f"{'transformer'}__{time_now}"

        wandb.init(
            project="transformer",
            entity="long-x",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
                       )
    run(total_epoch=epoch, best_loss=inf, track = args.track)
