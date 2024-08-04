import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
import json
import clip
import pandas as pd
from transformers import AutoTokenizer


class VideoQADataset(Dataset):
    def __init__(self, dataset_name, question_pt, question_type, num_frames, split, tokenizer):
        super(VideoQADataset, self).__init__()
        assert dataset_name in ["msvd-qa", "msrvtt-qa", "tgif-qa"]
        # self.features_dir = features_dir
        self.dataset_name = dataset_name
        self.question_pt = question_pt
        self.question_type = question_type
        self.tokenizer = tokenizer
        if self.question_type in ['action', 'transition']:
            self.ans_candidates = [0]
            self.ans_candidates_len = [0]

        # self.cand_answer_token = None
        # if split == 'train':
        # ## 计算acc
        #     with open('/root/autodl-tmp/code/DiffVQA/data/msrvtt-qa/msrvtt-qa_answers_list.json' ,'r') as f:
        #         answer2id = json.load(f)
        #         cand_answer = list(answer2id)
        #     cand_answer_token = tokenizer(cand_answer, padding="max_length", max_length=8, return_tensors='pt')
        #     for i in cand_answer_token:
        #         cand_answer_token[i] = cand_answer_token[i].cuda()
        #     self.cand_answer_token = cand_answer_token
        # answer_feat.append(model.lm(**answer_token).last_hidden_state.mean(1)) # 1, d
        # cand_answer = torch.stack(answer_feat, dim=0) # 4000, d
        # self.num_frames = num_frames

        # print('loading features from: \n %s \n %s' % (features_dir[0], features_dir[1]))


        # print('loading %d samples from %s' % (len(questions), question_pt))



        # 得到视频预提取的特征
        ## msvd
        ## msrvtt
        ## tgif
        # video_name --> feat
        # import pdb; pdb.set_trace()

        # 得到问题

        # 得到 tokenizer
        # from transformers import RobertaTokenizer
        # self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/roberta-large')
        # from transformers import AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/clip-vit-large-patch14')
        

        # ## 得到 id2caption msvd
        # ## msrvtt
        # ## tgif

        # tgif
        # self.key2name = {}
        # name_list = list(data_csv['gif_name'])
        # key_list = list(data_csv['key'])
        # for key, name in zip(key_list, name_list):
        #     self.key2name[key] = name


        # tgif  question
        # self.key2sentences = {}
        # for key, quest in zip(key_list, question_list):
        #     self.key2sentences[key] = quest


        # import pdb; pdb.set_trace()
        if self.dataset_name == 'tgif-qa':
            ### 
            self.videos_feats = torch.load('/root/autodl-tmp/data/tgif_frameqa/id2feat_{}.pt'.format(split))
            ### 
            # path2caption = json.load(open('/root/autodl-tmp/data/tgif_frameqa/path2caption_otter_multi_tgif.json', 'r'))
            # self.name2caption = {}
            # for path in list(path2caption.keys()):
            #     name = path.split('/')[-1].split('.')[0]
            #     self.name2caption[name] = path2caption[path]
            ###
            data_csv = pd.read_csv('/root/autodl-tmp/data/tgif_frameqa/{}_frameqa_question.csv'.format(split), sep='\t')
            self.name_list = list(data_csv['gif_name'])
            self.answers = list(data_csv['answer'])
            with open('/root/autodl-tmp/code/DiffVQA/data/tgif-qa/frameqa/tgif-qa_frameqa_answers_list.json', 'r') as f:
                ans2id = json.load(f)
            self.answers = [ans2id[ans] if ans in ans2id else 0 for ans in self.answers]
            # print("Total num: ", len(self.answers))
            # print("NULL num: ", sum([ans==0 for ans in self.answers]))
            ###
            self.question_list = list(data_csv['question'])
            self.answers_str = list(data_csv['answer'])
            # max_length = 0
            # for answer_str in self.answers_str:
            #     answer_token = self.tokenizer(answer_str, return_tensors='pt')['input_ids'].squeeze(0)
            #     max_length = max(max_length, answer_token.shape[0])
        elif self.dataset_name == 'msvd-qa':
            with open(question_pt, 'rb') as f:
                obj = pickle.load(f)
                # questions = obj['questions']
                # questions_len = obj['questions_len']
                video_ids = obj['video_ids'] # key 和 video name 是一回事
                answers = obj['answers']
                questions_word = obj['questions_word']
                ans_candidates = np.zeros(5)
                ans_candidates_len = np.zeros(5)
                if question_type in ['action', 'transition']:
                    ans_candidates = obj['ans_candidates']
                    ans_candidates_len = obj['ans_candidates_len']
            # self.questions = torch.LongTensor(np.asarray(questions))
            # self.questions_len = torch.LongTensor(np.asarray(questions_len))
            self.video_ids = video_ids
            self.answers = answers
            self.questions_word = questions_word

            if self.question_type in ['action', 'transition']:
                self.ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
                self.ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))
            ###
            self.videos_feats = torch.load('/root/autodl-tmp/data/msvd/id2feat.pt')
            ###
            self.sentences = []
            self.answers_str = []
            with open('/root/autodl-tmp/data/msvd/{}_qa.json'.format(split), 'r') as f:
                val = json.load(f)
                for qa in val:
                    self.sentences.append(qa['question'])
                    self.answers_str.append(qa['answer'])
            ###
            # path2caption = json.load(open('/root/autodl-tmp/data/msvd/path2caption_otter_multi.json', 'r'))
            # name2id = {line.split(' vid')[0]:int(line.split(' vid')[1].split('\n')[0]) 
            #         for line in open('/root/autodl-tmp/data/msvd/youtube_mapping.txt', 'r').readlines()}
            # self.id2caption = {}
            # for path in list(path2caption.keys()):
            #     name = path.split('/')[-1].split('.')[0]
            #     self.id2caption[name2id[name]] = path2caption[path]
        elif self.dataset_name == 'msrvtt-qa':
            with open(question_pt, 'rb') as f:
                obj = pickle.load(f)
                # questions = obj['questions']
                # questions_len = obj['questions_len']
                video_ids = obj['video_ids'] # key 和 video name 是一回事
                answers = obj['answers']
                questions_word = obj['questions_word']
                ans_candidates = np.zeros(5)
                ans_candidates_len = np.zeros(5)
                if question_type in ['action', 'transition']:
                    ans_candidates = obj['ans_candidates']
                    ans_candidates_len = obj['ans_candidates_len']
            # self.questions = torch.LongTensor(np.asarray(questions))
            # self.questions_len = torch.LongTensor(np.asarray(questions_len))
            self.video_ids = video_ids
            self.answers = answers
            self.questions_word = questions_word

            if self.question_type in ['action', 'transition']:
                self.ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
                self.ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))
            ###
            self.videos_feats = torch.load('/root/autodl-tmp/data/msrvtt/id2feat_TrainVal.pt')
            self.videos_feats.update(torch.load('/root/autodl-tmp/data/msrvtt/id2feat_Test.pt'))
            ###
            self.sentences = []
            self.answers_str = []
            with open('/root/autodl-tmp/data/msrvtt/{}_qa.json'.format(split), 'r') as f:
                val = json.load(f)
                for qa in val:
                    self.sentences.append(qa['question'])
                    self.answers_str.append(qa['answer'])
            ###
            # path2caption = json.load(open('/root/autodl-tmp/data/msrvtt/path2caption_otter_multi.json', 'r'))
            # self.id2caption = {}
            # for path in list(path2caption.keys()):
            #     id = int(path.split('/')[-1].split('.')[0].split('video')[-1])
            #     self.id2caption[id] = path2caption[path]
        # import pdb; pdb.set_trace()
            




        
        

        


    def __getitem__(self, item):
        # item = 12419
        # item = 4191
        # item = 725
        # item = 3334
        # question = self.questions[item]
        # question_len = self.questions_len[item]
        # tgif
        # video_name = self.key2name[video_id]
        answer = self.answers[item]

        if self.question_type in ['action', 'transition']:
            ans_candidate = self.ans_candidates[item]
            ans_candidate_len = self.ans_candidates_len[item]
        else:
            ans_candidate = torch.zeros(5)
            ans_candidate_len = torch.zeros(5)
        
        if self.dataset_name == 'tgif-qa':  
            question_word = self.question_list[item].split()[0]      
            video_name = self.name_list[item]
            # print(video_name)
            ###
            video = self.videos_feats[video_name].float() # 16,  768
            if video.shape[0] < 16:
                for i in range(16-video.shape[0]):
                    video = torch.cat([video, video[-1].clone().unsqueeze(0)], dim=0)
            assert video.shape[0] == 16 and video.shape[1] == 768, print(video.shape)
            ###
            question = self.tokenizer(self.question_list[item], padding="max_length", max_length=20, truncation=True, return_tensors='pt')
            for i in question:
                question[i] = question[i].squeeze(0)
            ###
            # context = self.tokenizer(self.name2caption[video_name], padding="max_length", max_length=36, truncation=True, return_tensors='pt')
            # for i in context:
            #     context[i] = context[i].squeeze(0)
            # print(item)
            # print(self.question_list[item])
            # print(question)
        elif self.dataset_name == 'msvd-qa':
            question_word = self.questions_word[item]
            video_id = self.video_ids[item]
            ### 
            video = self.videos_feats[video_id].float() # 16,  768
            if video.shape[0] < 16:
                for i in range(16-video.shape[0]):
                    video = torch.cat([video, video[-1].clone().unsqueeze(0)], dim=0)
            assert video.shape[0] == 16 and video.shape[1] == 768, print(video.shape)
            ###
            question = self.tokenizer(self.sentences[item], padding="max_length", max_length=20, truncation=True, return_tensors='pt')
            for i in question:
                question[i] = question[i].squeeze(0)
            ###
            # context = self.tokenizer(self.id2caption[video_id], padding="max_length", max_length=36, truncation=True, return_tensors='pt')
            # for i in context:
            #     context[i] = context[i].squeeze(0)
        elif self.dataset_name == 'msrvtt-qa':
            question_word = self.questions_word[item]
            video_id = self.video_ids[item]
            ### 
            video = self.videos_feats[video_id].float() # 16,  768
            if video.shape[0] < 16:
                for i in range(16-video.shape[0]):
                    video = torch.cat([video, video[-1].clone().unsqueeze(0)], dim=0)
            assert video.shape[0] == 16 and video.shape[1] == 768, print(video.shape)
            ###
            question = self.tokenizer(self.sentences[item], padding="max_length", max_length=20, truncation=True, return_tensors='pt')
            for i in question:
                question[i] = question[i].squeeze(0)
            ###
        answer_token = self.tokenizer(self.answers_str[item], padding="max_length", max_length=8, return_tensors='pt')
        for i in answer_token:
            answer_token[i] = answer_token[i].squeeze(0)
            # assert answer_token.shape[0] == 3, print(self.answers_str[item], answer_token)
            # context = self.tokenizer(self.id2caption[video_id], padding="max_length", max_length=36, truncation=True, return_tensors='pt')
            # for i in context:
            #     context[i] = context[i].squeeze(0)
                
        
        # get context from question
        # items = [i for i, vid in enumerate(self.video_ids) if vid==video_id]
        # context = torch.cat([torch.tensor(self.questions[i]) for i in items], dim=0).mean(dim=0, keepdim=True) # 1, d
        # context = torch.tensor(self.videos_caption[video_id].mean(dim=0, keepdim=True)) # 1, d
        # context = torch.tensor(self.videos_caption[video_id]) # 20, d
        # context = torch.randn_like(context)
         
        return (answer_token, video, question, ans_candidate, ans_candidate_len, answer, question_word)

    def __len__(self):
        return len(self.answers)
    
    def get_cand(self):
        # return self.cand_answer_token # 4000, 8
        return None
