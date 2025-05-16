import numpy as np 
import torch
import torch.nn as nn
import copy
from torchvision import transforms
from PIL import Image
import pickle
import pdb
import torch.nn.functional as F

class SGAttacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model=model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    
    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        # original state
        with torch.no_grad():
            # origin_img_output是一个字典包含image_feats和image_embed，其中image_embed是encoder的输出结果torch.Size([2, 577, 768])，image_feats是image_embed的cls token嵌入结果再过一下投影层torch.Size([2, 256])
            origin_img_output = self.model.Gen_image_feats(self.img_attacker.normalization(imgs))
            #当你使用 txt2img = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 作为索引时，你实际上是在说：“给我所有索引为0的行和索引为1的行，每个都重复一次。”相当于对每一个句子都找到与其对应的图片的特征
            img_supervisions = origin_img_output['image_feat'][txt2img] 
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,txt2img=txt2img)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.Gen_text_feats(txts_input)
            txt_supervisions = txts_output['text_feat']
            #===========================原始文本======================
            ori_txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
            ori_txts_output = self.model.Gen_text_feats(ori_txts_input)
            ori_txt_supervisions = ori_txts_output['text_feat']
            #===========================原始图像============================
            ori_imgs_output = self.model.Gen_image_feats(self.img_attacker.normalization(imgs))
            ori_imgs_feats_list = ori_imgs_output['img_feats_list']
        # momentum = torch.zeros_like(imgs).detach().to(device)
        # momentum = self.img_attacker.pre_attack(self.model, imgs, txt2img, device, 
        #                                                scales=scales, txt_embeds = txt_supervisions,momentum=momentum,steps=1)
        adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, device, 
                                                       scales=scales, txt_embeds = txt_supervisions, ori_imgs_feats=ori_imgs_feats_list,ori_texts_embeds=ori_txt_supervisions)#,momentum = momentum
        # with torch.no_grad():
        #     adv_imgs_outputs = self.model.Gen_image_feats(self.img_attacker.normalization(adv_imgs))
        #     img_supervisions = adv_imgs_outputs['image_feat']
        #     txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
        #     txts_output = self.model.Gen_text_feats(txts_input)
        #     txt_supervisions = txts_output['text_feat']
        #     matrix_i2t = img_supervisions@txt_supervisions.T
        #     t2i_scores_ori = []
        #     for txt_id in range(len(txt2img)):
        #         t2i_scores_ori.append(matrix_i2t[txt2img[txt_id]][txt_id])
        
        # with torch.no_grad():
        #     adv_imgs_outputs = self.model.Gen_image_feats(self.img_attacker.normalization(adv_imgs))
        #     img_supervisions = adv_imgs_outputs['image_feat']#[txt2img]
        # adv_txts = self.txt_attacker.img_guided_attack_second(self.model, txts, img_embeds=img_supervisions,txt2img=txt2img,ori_t2i_scores=t2i_scores_ori,adv_txts=adv_txts)
                            
        return adv_imgs, adv_txts

                

class ImageAttacker():
    def __init__(self, normalization, eps=2/255, steps=10, step_size=0.5/255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps 
        self.step_size = step_size 
        self.decay = 0.9
        self.alpha = 1/self.steps
    #这个其实就是改版的ITC
    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        for i in range(len(txt2img)):
            #第txt2img[i]号图片与i号文本是1，代表匹配
            it_labels[txt2img[i], i]=-10
        
        loss_IaTcpos = (it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss
    
    def loss_layer(self, adv_img_feat, origin_img_feat, importance_s, device):
        cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        n = len(adv_img_feat)
        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
        for i in range(n):
            cos_s = cosine_similarity(adv_img_feat[i], origin_img_feat[i])  #torch.Size([2, 577]),(batch_size, 577)
            #把每一次的权重乘进去
            mean_value = importance_s[i] * torch.mean(cos_s,dim=1)
            #==================消融实验=============================
            # mean_value = torch.mean(cos_s,dim=1)
            loss += torch.mean(mean_value)
        return loss

    def get_layer_important_score(self, ori_imgs_feats):
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        importance_score = []
        final_feats = ori_imgs_feats[-1][:,0,:]
        for layer_feats in ori_imgs_feats:
            layer_f_cls = layer_feats[:,0,:]
            importance_score.append(cosine_similarity(layer_f_cls, final_feats))
        importance_score = torch.stack(importance_score, dim=0)
        # 计算最小值和最大值
        min_val = torch.min(importance_score)
        max_val = torch.max(importance_score)

        # 执行最小-最大归一化
        #importance_score = 0.9 * (importance_score - min_val) / (max_val - min_val) + 0.1
        importance_score = (importance_score - min_val) / (max_val - min_val)
        return importance_score
    
    def loss_feature_space_ori(self, adv_img_feat, ori_text_feats, txt2img):

        device = adv_img_feat.device    
        #============将图像特征与原始文本特征拉远===============
        it_sim_matrix = adv_img_feat @ ori_text_feats.T 
        b = it_sim_matrix.shape[0]
        result_matrix = torch.zeros(b, 2).to(device)
        print(it_sim_matrix[0])
        # 遍历每个图片
        for img_idx in range(b):
            # 获取当前图片对应的文本索引
            correspond_indices = [index for index, value in enumerate(txt2img) if value == img_idx]
            not_correspond_indices = [index for index, value in enumerate(txt2img) if value != img_idx]
            # 计算当前图片与对应文本的相似度和
            corresponding_sum = torch.sum(it_sim_matrix[img_idx][correspond_indices])
            not_corresponding_sum = torch.sum(it_sim_matrix[img_idx][not_correspond_indices])
            # # 将结果存储到结果矩阵中
            result_matrix[img_idx, 0] = corresponding_sum
            result_matrix[img_idx, 1] = not_corresponding_sum
        
        labels = []
        for i in range(b):
            labels.append(1)
        labels = torch.tensor(labels).to(device)  # 第一个样本属于类别0和类别2
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(result_matrix, labels) 
        
        return loss
    def loss_feature_space(self, adv_img_feat, ori_text_feats, txt2img):  
        device = adv_img_feat.device    
        #============将图像特征与原始文本特征拉远===============
        it_sim_matrix = adv_img_feat @ ori_text_feats.T 
        # print(it_sim_matrix[0])
        it_labels = torch.ones(it_sim_matrix.shape).to(device)
        #it_labels = it_labels * self.alpha#(1-self.alpha)
        for i in range(len(txt2img)):
            #第txt2img[i]号图片与i号文本是1，代表匹配
            it_labels[txt2img[i], i]=-10#-15#*self.alpha
        
        loss_IaTcpos = (it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        #print(f'alpha:{self.alpha}')
        
        return loss

    def pre_attack(self, model, imgs, txt2img, device, steps,momentum,scales=None, txt_embeds=None):
        
        model.eval()
       
        b, _, _, _ = imgs.shape#torch.Size([2, 3, 384, 384])
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        #==============初始化动量================
        #momentum = torch.zeros_like(imgs).detach().to(device)
        self.alpha = 1/steps
        for i in range(steps):
            adv_imgs.requires_grad_()
            model.zero_grad()
            scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device) #torch.Size([10, 3, 384, 384])假如batchsize=2，scales有4种最终生成10张图片,形成的是一个batch一个batch的。也就是0号图片的第一个形变图片在batch号，第二个在batch*2号  
            if self.normalization is not None:
                adv_imgs_output = model.Gen_image_feats(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.Gen_image_feats(scaled_imgs)
                
            adv_imgs_embeds = adv_imgs_output['image_feat']#torch.Size([10, 256])与上面对应

            with torch.enable_grad():
                loss_list = []
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    loss_item = self.loss_feature_space(adv_imgs_embeds[i*b:i*b+b], txt_embeds, txt2img)
                    #loss_item = self.loss_func(adv_imgs_embeds[i*b:i*b+b], txt_embeds, txt2img)
                    loss_list.append(loss_item.item())
                    loss += loss_item
            loss.backward()
            #==============尝试引入动态===========================
            #self.alpha += 1/self.steps
            grad = adv_imgs.grad 

            #momentum2 = self.get_momentum(grad, momentum2)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)     
            grad = grad+momentum * 0.9
            momentum = grad      
            perturbation = self.step_size * grad.sign()
            #perturbation = self.step_size * momentum2.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        return momentum
    
    def txt_guided_attack(self, model, imgs, txt2img, device, ori_imgs_feats, scales=None, txt_embeds=None,ori_texts_embeds=None):
        
        model.eval()
       
        b, _, _, _ = imgs.shape#torch.Size([2, 3, 384, 384])
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        #=========获取重要性==========
        layer_importance = self.get_layer_important_score(ori_imgs_feats)
        #==============初始化动量================
        #momentum = torch.zeros_like(imgs).detach().to(device)
        self.alpha = 1/self.steps
        #print(self.steps)
        for i in range(self.steps):
            adv_imgs.requires_grad_()
            #==================测试加入变化效果====================
            # scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device) #torch.Size([10, 3, 384, 384])假如batchsize=2，scales有4种最终生成10张图片,形成的是一个batch一个batch的。也就是0号图片的第一个形变图片在batch号，第二个在batch*2号  
        
           
            
            model.zero_grad()
            #=============层间攻击==========================
            adv_imgs_output = model.Gen_image_feats(self.normalization(adv_imgs))
            adv_imgs_feats_list = adv_imgs_output['img_feats_list']
            adv_imgs_embeds = adv_imgs_output['image_feat']
            with torch.enable_grad():
                # for i in range(scales_num):
                loss_item = self.loss_layer(adv_imgs_feats_list, ori_imgs_feats, layer_importance, device)
                # loss_ori_txt = self.loss_func(adv_imgs_embeds,txts_embeds=ori_texts_embeds,txt2img=txt2img)
                loss = loss_item#+loss_ori_txt
            loss.backward()            
            grad = adv_imgs.grad 
            #momentum1 = self.get_momentum(grad, momentum1)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            perturbation = self.step_size * grad.sign()
            #perturbation = self.step_size * momentum1.sign()
            adv_imgs = adv_imgs.detach() - perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            
            adv_imgs.requires_grad_()
            scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device) #torch.Size([10, 3, 384, 384])假如batchsize=2，scales有4种最终生成10张图片,形成的是一个batch一个batch的。也就是0号图片的第一个形变图片在batch号，第二个在batch*2号  
        
            if self.normalization is not None:
                adv_imgs_output = model.Gen_image_feats(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.Gen_image_feats(scaled_imgs)
                
            adv_imgs_embeds = adv_imgs_output['image_feat']#torch.Size([10, 256])与上面对应

            with torch.enable_grad():
                loss_list = []
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    loss_item = self.loss_feature_space(adv_imgs_embeds[i*b:i*b+b], txt_embeds, txt2img)
                    loss_ori_txt = self.loss_func(adv_imgs_embeds[i*b:i*b+b],txts_embeds=ori_texts_embeds,txt2img=txt2img)
                    #loss_item = self.loss_func(adv_imgs_embeds[i*b:i*b+b], txt_embeds, txt2img)
                    loss_list.append(loss_item.item())
                    loss += loss_item
                    loss += loss_ori_txt
            loss.backward()
            #==============尝试引入动态===========================
            grad = adv_imgs.grad 

            #momentum2 = self.get_momentum(grad, momentum2)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)  
            #grad = grad+momentum * 0.9
            #momentum = grad         
            perturbation = self.step_size * grad.sign()
            #perturbation = self.step_size * momentum2.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        return adv_imgs



    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)



filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
    

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32):#device,embeddings,
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls#True
        self.idx2word = {}
        self.word2idx = {}
        self.sim_lis = []
        self.word_vector()
    
    def word_vector(self):
        print("Building vocab...")
        with open('', 'r', encoding='UTF-8') as ifile:
            i = 0
            for line in ifile:
                i+=1
                word = line.split()[0]
                if word not in self.idx2word:
                    self.idx2word[len(self.idx2word)] = word
                    self.word2idx[word] = len(self.idx2word) - 1
        print(f"Counter_fitting中有{i}个单词")

        print("Building cos sim matrix...")
        # load pre-computed cosine similarity matrix if provided
        print(f'Load pre-computed cosine similarity matrix from mat.txt')
        with open('', "rb") as fp:
            self.sim_lis = pickle.load(fp)
        print("Cos sim import finished!")
        #塞入嵌入
        # self.embeddings = embeddings#embeddings=model.text_encoder.embeddings.to(device)
        # self.device = device
        # self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
        #                                                    do_lower_case="uncased" in "bert-base-uncased")
        
   
    def img_guided_attack(self, net, texts, txt2img, img_embeds = None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)   # seq-len k shape:(len(texts),30,10),每个句子分成30个单词，每个单词10个预测。

        # original state
        origin_output = net.Gen_text_feats(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):#该层循环实现句子替换
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)#得到每一个单词的重要性，因为在dataset提取的时候已经

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, sub_words, keys = self._tokenize(text)#words就是句子里的单词没有标点了已经。
            final_words = copy.deepcopy(words)
            change = 0
            substitute_word = None
            substitute_word_score = -10
            word_idx = 0
            for top_index in list_of_index:#该层循环实现每个句子中单词的替换
                
                if change >= self.num_perturbation:
                    break
                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue
                if words[top_index[0]] in self.word2idx.keys():
                    substitutes = []
                    word_vector_idx = self.word2idx[words[top_index[0]] ]
                    sub_words_list_idx = self.sim_lis[word_vector_idx]#[1:11]
                    for sub_index in sub_words_list_idx:
                        if(sub_index[0]>0.4):
                            substitutes.append(self.idx2word[sub_index[1]])#得到符合条件的替换词
                else:
                    substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k. word_predictions是一个len(texts)*30*10的矩阵，这样抽取出替换的token（不是单词）
                    #tensor([[1037, 1996, 1998, 1000, 1997, 1012, 2019, 2009, 1999, 2178],[2450, 2158, 2273, 2308, 2299, 3232, 1998, 6045, 2040, 1010]],device='cuda:0')
                    word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                    substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                                    self.threshold_pred_score)
                #substitutes:['a', 'the', 'some', 'an', 'any', 'another', 'one', 'every', 'it', 'each'],代表这个位置的替换词。

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:#每个句子中的每个具体位置替换
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    # if substitute in self.word2idx.keys() and tgt_word in self.word2idx.keys():
                    #     t = 0
                    #     ori_idx = self.word2idx[tgt_word]
                    #     tar_idx = self.word2idx[substitute]
                    #     sub_words_list_idx = self.sim_lis[ori_idx]#[1:11]
                    #     for sub_index in sub_words_list_idx:
                    #         if(sub_index[1]==tar_idx):
                    #             t=1
                    #             break
                    #     if t == 0:
                    #         continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.Gen_text_feats(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, img_embeds, i)#
                #loss = self.loss_func_second(replace_embeds, img_embeds, i, txt2img)
                #candidate_idx = loss.argmin()
                candidate_idx = loss.argmax()
                #找到最大的句子并替换，如果不是原句子就change+1
                # if(loss[candidate_idx]<substitute_word_score):
                #     substitute_word_score = loss[candidate_idx]
                #     substitute_word = available_substitutes[candidate_idx]
                #     word_idx = top_index[0]
                if loss[candidate_idx]>substitute_word_score :
                    substitute_word_score = loss[candidate_idx]
                    substitute_word = available_substitutes[candidate_idx]
                    word_idx = top_index[0]
            final_words[word_idx] = substitute_word
            if available_substitutes[candidate_idx] != tgt_word:
                change += 1
            final_adverse.append(' '.join(final_words))

        return final_adverse

    def img_guided_attack_second(self, net, texts, txt2img,ori_t2i_scores,adv_txts,img_embeds = None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)   # seq-len k shape:(len(texts),30,10),每个句子分成30个单词，每个单词10个预测。

        # original state
        origin_output = net.Gen_text_feats(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):#该层循环实现句子替换
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)#得到每一个单词的重要性，因为在dataset提取的时候已经

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, sub_words, keys = self._tokenize(text)#words就是句子里的单词没有标点了已经。
            final_words = copy.deepcopy(words)
            use_ori_adv = True
            substitute_word = None
            substitute_word_score = 0
            word_idx = 0
            for top_index in list_of_index:#该层循环实现每个句子中单词的替换
                
                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue
                if words[top_index[0]] in self.word2idx.keys():
                    substitutes = []
                    word_vector_idx = self.word2idx[words[top_index[0]] ]
                    sub_words_list_idx = self.sim_lis[word_vector_idx]#[1:11]
                    for sub_index in sub_words_list_idx:
                        if(sub_index[0]>0.4):
                            substitutes.append(self.idx2word[sub_index[1]])#得到符合条件的替换词
                else:
                    substitutes = word_predictions[i, keys[top_index[0]][0]+1:keys[top_index[0]][1]+1]  # L, k. word_predictions是一个len(texts)*30*10的矩阵，这样抽取出替换的token（不是单词）
                    #tensor([[1037, 1996, 1998, 1000, 1997, 1012, 2019, 2009, 1999, 2178],[2450, 2158, 2273, 2308, 2299, 3232, 1998, 6045, 2040, 1010]],device='cuda:0')
                    word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]+1:keys[top_index[0]][1]+1]

                    substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                                    self.threshold_pred_score)
                #substitutes:['a', 'the', 'some', 'an', 'any', 'another', 'one', 'every', 'it', 'each'],代表这个位置的替换词。

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:#每个句子中的每个具体位置替换
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                #==================添加原始adv
                replace_texts.append(adv_txts[i])
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.Gen_text_feats(replace_text_input)
                
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func_second(replace_embeds, img_embeds, i, txt2img,ori_t2i_scores=ori_t2i_scores)
                candidate_idx = loss.argmax()
                # loss = self.loss_func(replace_embeds, img_embeds, i)
                # candidate_idx = loss.argmax()
                #找到最大的句子并替换，如果不是原句子就change+1
                
                if(loss[candidate_idx]>substitute_word_score):
                    print(f'第{i}句')
                    if candidate_idx == len(replace_texts)-1:
                        use_ori_adv = True 
                    else:
                        substitute_word_score = loss[candidate_idx]
                        substitute_word = available_substitutes[candidate_idx]
                        word_idx = top_index[0]
                        use_ori_adv = False
            if use_ori_adv:
                final_adverse.append(adv_txts[i])
            else:
                final_words[word_idx] = substitute_word
                final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1) #虽然是点乘和内积是一样的，加负号就是为了找到最小的相似度，因为相似度越大变负号越小
        #这里用到了逐元素乘法，实际算出来和矩阵乘获得相似度是一样的
        loss = loss_TaIcpos
        return loss
    def loss_func_second(self, txt_embeds, img_embeds, label,txt2img,ori_t2i_scores):
        it_sim_matrix = txt_embeds @ img_embeds.T
        img_idx = txt2img[label]
        it_sim_matrix[:,img_idx]=it_sim_matrix[:,img_idx] * -15
        loss = it_sim_matrix.sum(-1)
        return loss
    # def loss_func_second(self, txt_embeds, img_embeds, label,txt2img,ori_t2i_scores):
    #     it_sim_matrix = txt_embeds @ img_embeds.T
    #     img_idx = txt2img[label]
    #     t2i_scores = it_sim_matrix[:,img_idx]
    #     softmax_scores = F.softmax(it_sim_matrix, dim=1)[:,img_idx]
    #     threshold = ori_t2i_scores[label]
    #     indices = torch.where(t2i_scores > threshold)[0]#找到所有大于原始值的下标后续将对应结果变成1
    #     softmax_scores[indices]=1
    #     return softmax_scores 
    # def loss_func_second(self, txt_embeds, img_embeds, label,txt2img):
    #     it_sim_matrix = txt_embeds @ img_embeds.T
    #     img_idx = txt2img[label]
    #     softmax_scores = F.softmax(it_sim_matrix, dim=1)[:,img_idx]
    #     return softmax_scores


    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k shape:(len(texts),30,10),每个句子分成30个单词，每个单词10个预测。

        # original state
        origin_output = net.Gen_text_feats(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()



        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length) 

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]+1:keys[top_index[0]][1]+1]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)
                

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.Gen_text_feats(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1), origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))
                
                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

 
    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)##这一步生成的是一系列word需要下面的代码将他们组合成句子
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        """
        # ['[UNK] man with pierced ears is wearing glasses and an orange hat',
        # 'the [UNK] with pierced ears is wearing glasses and an orange hat',
        # 'the man [UNK] pierced ears is wearing glasses and an orange hat',
        # 'the man with [UNK] ears is wearing glasses and an orange hat',
        # 'the man with pierced [UNK] is wearing glasses and an orange hat',
        # 'the man with pierced ears [UNK] wearing glasses and an orange hat',
        # 'the man with pierced ears is [UNK] glasses and an orange hat',
        # 'the man with pierced ears is wearing [UNK] and an orange hat',
        # 'the man with pierced ears is wearing glasses [UNK] an orange hat',
        # 'the man with pierced ears is wearing glasses and [UNK] orange hat',
        # 'the man with pierced ears is wearing glasses and an [UNK] hat',
        # 'the man with pierced ears is wearing glasses and an orange [UNK]']
        """
        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.Gen_text_feats(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()#masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)



def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
