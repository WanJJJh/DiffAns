import torch
import numpy as np
import argparse
import logging
import time
import os
import pickle
import json
from tqdm import tqdm
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from DataLoader import VideoQADataset
from network import VideoQANetwork, V_CAT, VQA
from validate import Validate
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        # nn.init.normal_(m.weight, 0.1)
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        else:
            print(m)

def contrastive_criterion(logits_v, logits_t):
    labels = torch.arange(logits_v.shape[0], device=logits_v.device).long()
    loss = (
        F.cross_entropy(logits_v, labels) + 
        F.cross_entropy(logits_t, labels)
    ) / 2
    return loss


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def main(args):

    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)

    # set data path
    features_dir = []
    if args.dataset == 'tgif-qa':
        for feat_type in args.features_type:
            feat_path = args.features_path.format(
                args.dataset, args.question_type, args.dataset, args.question_type, feat_type)
            features_dir.append(feat_path)
        question_pt_train = args.question_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type, 'train')
        question_pt_test = args.question_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type, 'test')
        answers_list_json = args.answers_list_json.format(
            args.dataset, args.question_type, args.dataset, args.question_type)
        glove_matrix_pt = args.glove_matrix_pt.format(
            args.dataset, args.question_type, args.dataset, args.question_type)
        save_dir = args.save_dir.format(args.dataset, args.question_type)
    else:
        for feat_type in args.features_type:
            feat_path = 'data/{}/{}_{}_feat'.format(
                args.dataset, args.dataset, feat_type)
            features_dir.append(feat_path)
        question_pt_train = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'train')
        question_pt_val = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'val')
        question_pt_test = 'data/{}/{}_{}_questions.pt'.format(
            args.dataset, args.dataset, 'test')
        answers_list_json = 'data/{}/{}_answers_list.json'.format(
            args.dataset, args.dataset)
        glove_matrix_pt = 'data/{}/{}_glove_matrix.pt'.format(
            args.dataset, args.dataset)
        save_dir = 'results/exp_{}'.format(args.dataset)

    lctime = time.localtime()
    lctime = time.strftime("%Y-%m-%d_%A_%H:%M:%S", lctime)
    fileHandler = logging.FileHandler(os.path.join(save_dir, lctime + '_stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/{}'.format(args.lm_name))
    # bert_file = '/root/autodl-fs/roberta-base/{}.h5'
    # if args.train:
    train_data = VideoQADataset(args.dataset, question_pt_train, args.question_type, args.num_frames, "train", tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    if args.val:
        val_data = VideoQADataset(args.dataset, question_pt_val, args.question_type, args.num_frames, "val", tokenizer)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    if args.test:
        test_data = VideoQADataset(args.dataset, question_pt_test, args.question_type, args.num_frames, "test", tokenizer)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)


    if args.question_type in ['action', 'count', 'transition']:
        num_answers = 1
    else:
        logging.info('load answers list')
        with open(answers_list_json, 'r') as f:
            answers_list = json.load(f)
            num_answers = len(answers_list)

    device = torch.device("cuda")
    GCN_adj, GAT_adj = None, None
    # logging.info('load glove vectors')
    # with open(glove_matrix_pt, 'rb') as f:
    #     glove_matrix = pickle.load(f)
    # glove_matrix = torch.FloatTensor(glove_matrix)
    # vocab_size = glove_matrix.shape[0]
    if not args.train:
        model_path = os.path.join('/root/autodl-tmp/code/DiffVQA/results', 'model_{}_{}.pth'.format(args.dataset, args.T))
        args, model_dict = load_model(model_path)
    # import pdb; pdb.set_trace()
    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'module_dim': args.module_dim,
        'question_type': args.question_type,
        'num_answers': num_answers,
        'dropout': args.dropout,
        'GCN_adj': GCN_adj,
        'GAT_adj': GAT_adj,
        "lm_frozen": args.lm_frozen,
        "T": args.T,
        'lm_name': args.lm_name,
        'scale': args.scale,
        'model_id': args.model_id,
        'layer_num': args.layer_num
    }
    model = VQA(**model_kwargs)
    # model.apply(weights_init)
    # with torch.no_grad():
    #     model.linguistic.encoder_embed.weight.set_(glove_matrix)
    if not args.train:
        model.load_state_dict(model_dict)
    model = model.to(device)
    if args.question_type == 'count':
        criterion = nn.MSELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    def criterion_L1(x, y):
        mse = nn.L1Loss(reduction='mean').cuda()
        out = mse(x, y)
        return out
    
    def criterion_SmoothL1(x, y):
        mse = nn.SmoothL1Loss(reduction='mean').cuda()
        out = mse(x, y)
        return out
    
    def criterion_L2(x, y):
        mse = nn.MSELoss(reduction='mean').cuda()
        out = mse(F.softmax(x, dim=-1), y)
        return out
    
    def criterion_consin(x, y, gt):
        # gt 代表 1 相似 -1 不相似
        cs = nn.CosineEmbeddingLoss(reduction='mean').cuda()
        out = cs(x, y, gt)
        return out

    def criterion_KL(x, y):
        x_log = F.log_softmax(x, dim=1)
        # 只转化为概率
        y = F.softmax(y, dim=1)
        kl = nn.KLDivLoss(reduction='batchmean').cuda()
        out = kl(x_log, y)
        return out

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.question_type in ['count', 'none']:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 19, 23, 27], gamma=0.5)
    if not args.train:
        Validate(args, model, test_loader, args.max_epochs, device, logging, "test", train_data.get_cand())
        return 0

    logging.info("Start training........")
    best_acc = 0.0
    best_mse = 1000.0
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        count = 0
        progress_bar = tqdm(train_loader)
        for idx, batch in enumerate(progress_bar):
            input_batch = list(map(lambda x: x.to(device), batch[:-1]))
            optimizer.zero_grad()
            answers = input_batch[-1] # 
            batch_size = answers.size(0)
            logits, mid_logits = model(*input_batch)
            if args.question_type in ['action', 'transition']:
                batch_agg = np.concatenate(np.tile(np.arange(batch_size).
                                                   reshape([batch_size, 1]),[1, 5])) * 5
                answers_agg = tile(answers, 0, 5)
                loss = torch.max(torch.tensor(0.0).cuda(),
                                 1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])

                loss = loss.mean()
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                aggreeings = (preds == answers)

            elif args.question_type == 'count':
                answers = answers.unsqueeze(-1)
                loss = criterion(logits, answers.float())
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2

            else:
                # import pdb; pdb.set_trace()
                # answers_onehot = F.one_hot(answers, num_answers).float()
                # import pdb; pdb.set_trace()
                #  
                loss = criterion(logits, answers) + criterion(mid_logits, answers)
                # + args.num_scale*contrastive_criterion(logits_v, logits_t)
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (idx + 1)

                optimizer.step()
                # import pdb; pdb.set_trace()
                # with torch.no_grad():
                #     cand_answer_token = train_data.get_cand()
                #     cand_answer_feat = model.lm(**cand_answer_token).last_hidden_state.mean(1) # 4000, 768
                #     # logits = torch.matmul(F.cosine_embedding_losssoftmax(logits, dim=-1), F.softmax(cand_answer_feat,dim=-1).T) # b, 4000
                #     logits_sim = F.cosine_similarity(logits.unsqueeze(1), cand_answer_feat.unsqueeze(0),dim=-1) # b, 4000
                    # 
                    # import pdb; pdb.set_trace()
                preds = logits.detach().argmax(1)
                aggreeings = (preds == answers)

            if args.question_type == 'count':
                total_mse += batch_mse.sum().item()
                count += batch_size
                progress_bar.set_description("Training epoch \033[1;33m{} \033[0m: loss: \033[1;34m {:.3f} \033[0m, avg_loss: \033[1;35m {:.3f} \033[0m, avg_mse: \033[1;31m {:.4f} \033[0m".format(
                                                    epoch + 1, loss.item(), avg_loss, total_mse / count))
            else:
                total_acc += aggreeings.sum().item()
                count += batch_size
                progress_bar.set_description("Training epoch \033[1;33m{} \033[0m: loss: \033[1;34m {:.3f} \033[0m, avg_loss: \033[1;35m {:.3f} \033[0m, avg_acc: \033[1;31m {:.4f} \033[0m".format(
                                                    epoch + 1, loss.item(), avg_loss, total_acc / count))
        scheduler.step()
        progress_bar.close()

        if args.val:
            if args.question_type == 'count':
                val_mse = Validate(args, model, val_loader, epoch, device, logging, val_type="val")
            else:
                if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                    epoch_accuracy_word, val_acc = Validate(args, model, val_loader, epoch, device, logging, val_type="val")
                else:
                    val_acc = Validate(args, model, val_loader, epoch, device, logging, val_type="val")

        if args.test:


            if args.question_type == 'count':
                test_mse = Validate(args, model, test_loader, epoch, device, logging, train_data.get_cand(), "test")
                if test_mse < best_mse:
                    best_mse = test_mse
                    # save_model(args, model, os.path.join(save_dir, 'model_{}.pth'.format(args.model_id)), best_mse)
            else:
                if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                    epoch_accuracy_word, test_acc = Validate(args, model, test_loader, epoch, device, logging, train_data.get_cand(), val_type="test")
                else:
                    test_acc = Validate(args, model, test_loader, epoch, device, logging, train_data.get_cand(), val_type="test")
                if test_acc > best_acc:
                    best_acc = test_acc
                    accuracy_word = None
                    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
                        accuracy_word = epoch_accuracy_word
                    # save_model(args, model, os.path.join('/root/autodl-tmp/code/DiffVQA/results', 'model_{}_{}.pth'.format(args.dataset, args.T)), best_acc, accuracy_word)

    logging.info('~~~~~Finall best results !~~~~~~~~~')
    if args.question_type == 'count':
        logging.info('Best MSE on {}_{} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, args.question_type, best_mse))
    else:
        if args.dataset in ['msvd-qa', 'msrvtt-qa']:
            logging.info('Best Question Words Accuracy: what {:.4f}, who {:.4f}, how {:.4f}, when {:.4f}, where {:.4f}'.format(
                    accuracy_word['what'], accuracy_word['who'], accuracy_word['how'], accuracy_word['when'], accuracy_word['where']))
            logging.info('Best Accuracy on {} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, best_acc))
        else:
            logging.info('Best Accuracy on {}_{} is: \033[1;31m {:.4f} \033[0m'.format(args.dataset, args.question_type, best_acc))
    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', default='data/{}/{}/{}_{}_{}_feat', type=str)
    parser.add_argument('--question_pt', type=str, default='data/{}/{}/{}_{}_{}_questions.pt')
    parser.add_argument('--glove_matrix_pt', type=str, default='data/{}/{}/{}_{}_glove_matrix.pt')
    parser.add_argument('--answers_list_json', type=str, default='data/{}/{}/{}_{}_answers_list.json')
    parser.add_argument('--save_dir', default='results/exp_{}_{}', type=str)
    parser.add_argument('--dataset', default='msvd-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--question_type', default='none',
                        choices=['action', 'count', 'frameqa', 'transition', 'none'], type=str)
    parser.add_argument('--features_type',
                        default=['appearance_pool5_16', 'motion_16'], type=str)

    # hyper-parameters
    # parser.add_argument('--lm_frozen', default=False, action='store_true')
    parser.add_argument('--lm_frozen', default=0, type=int)
    parser.add_argument('--lm_name', default="roberta-base", choices=['vit-l14', 'roberta-base', 'roberta-large', 'deberta-base', 'deberta_nli_base'], type=str)
    parser.add_argument('--T', default=3, type=int)
    parser.add_argument('--scale', default=1, type=float)

    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--module_dim', default=768, type=int)
    parser.add_argument('--app_pool5_dim', default=2048, type=int)
    parser.add_argument('--motion_dim', default=2048, type=int)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--max_epochs', default=25, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.000005, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--model_id', default=0, type=int)
    parser.add_argument('--layer_num', default=1, type=int)

    parser.add_argument('--use_train', default=False, dest='train', action='store_true')
    parser.add_argument('--use_val', default=False, dest='val', action='store_true')
    parser.add_argument('--use_test', default=False, dest='test', action='store_true')

    args = parser.parse_args()
    main(args)