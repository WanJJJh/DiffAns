import torch
import numpy as np
import os
from tqdm import tqdm


def Validate(args, model, val_loader, epoch, device, logging, cand_answer_token, val_type="val"):
    model.eval()
    total_acc = 0.0
    total_mse = 0.0
    count = 0

    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
        correct_word = {"what": 0, "who": 0, "how": 0, "when": 0, "where": 0}
        count_word = {"what": 0, "who": 0, "how": 0, "when": 0, "where": 0}
        accuracy_word = {"what": 0, "who": 0, "how": 0, "when": 0, "where": 0}

    progress_bar = tqdm(val_loader)
    for idx, batch in enumerate(progress_bar):
        # import pdb; pdb.set_trace()
        question_word = batch[-1]
        input_batch = list(map(lambda x: x.to(device), batch[:-1]))
        answers = input_batch[-1]
        batch_size = answers.size(0)
        with torch.no_grad():
            
            # cand_answer_feat = model.lm(**cand_answer_token).last_hidden_state.mean(1) # 4000, 768
            logits, _ = model(*input_batch[:-1]) # b, d
            # logits = logits+ logits_
            # logits = torch.matmul(logits, cand_answer_feat.T) # b, 4000

        if args.question_type in ['action', 'transition']:
            preds = torch.argmax(logits.view(batch_size, 5), dim=1)
            aggreeings = (preds == answers)
        elif args.question_type == 'count':
            answers = answers.unsqueeze(dim=-1)
            preds = (logits + 0.5).long().clamp(min=1, max=10)
            batch_mse = (preds - answers) ** 2
        else:
            preds = logits.detach().argmax(1)
            aggreeings = (preds == answers)
        # print(preds.item(), answers.item(), aggreeings.item())

        if args.question_type == 'count':
            total_mse += batch_mse.sum().item()
            count += batch_size
        else:
            total_acc += aggreeings.sum().item()
            count += batch_size
        if val_type == "val":
            progress_bar.set_description("Validating epoch \033[1;33m{} \033[0m".format(epoch + 1))
        else:
            progress_bar.set_description("Testing epoch \033[1;33m{} \033[0m".format(epoch + 1))

        if args.dataset in ['msvd-qa', 'msrvtt-qa']:
            for i in range(batch_size):
                count_word[question_word[i]] += 1
                correct_word[question_word[i]] += aggreeings.cpu().numpy()[i]

    progress_bar.close()

    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
        for word in accuracy_word:
            accuracy_word[word] = correct_word[word] / count_word[word]
        logging.info('Epoch {} Question Words Accuracy: what {:.4f}, who {:.4f}, how {:.4f}, when {:.4f}, where {:.4f}'.format(
            epoch + 1, accuracy_word['what'], accuracy_word['who'], accuracy_word['how'], accuracy_word['when'], accuracy_word['where']))

    if args.question_type == 'count':
        logging.info('Epoch \033[1;33m{} \033[0m {} MSE: \033[1;31m {:.4f} \033[0m'.format(epoch + 1, val_type, total_mse / count))
        return total_mse / count
    else:
        logging.info('Epoch \033[1;33m{} \033[0m {} Accuracy: \033[1;31m {:.4f} \033[0m'.format(epoch + 1, val_type, total_acc / count))
        if args.dataset in ['msvd-qa', 'msrvtt-qa']:
            return accuracy_word, total_acc / count
        else:
            return total_acc / count
