import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    # parser.add_argument('--config', dest='config', help='settings of APE in yaml format')
    parser.add_argument('--shot', dest='shot', type=int, default=4, help='shots number')
    parser.add_argument('--config',  default='./configs/imagenet.yaml', help='settings of APE in yaml format')
    args = parser.parse_args()
    return args


def APE(cfg, cache_keys, cache_keys_map, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):

    cache_keys = cache_keys.float().to(cfg['cuda'])
    cache_keys_map = cache_keys_map.float().to(cfg['cuda'])
    cache_values = cache_values.float().to(cfg['cuda'])
    val_features = val_features.float().to(cfg['cuda'])
    val_labels = val_labels.float().to(cfg['cuda'])
    test_features = test_features.float().to(cfg['cuda'])
    test_labels = test_labels.float().to(cfg['cuda'])
    clip_weights = clip_weights.float().to(cfg['cuda'])

    samp_num = cache_keys_map.shape[0]
    feat_dim, cate_num = clip_weights.shape
    cache_labels = torch.argmax(cache_values, -1)
    
    img_simi = torch.matmul(cache_keys_map.reshape(samp_num, -1, feat_dim), cache_keys.t()[:,None,:].permute(0,2,1)).squeeze()
    img_simi, img_simi_idx = torch.topk(img_simi, k=30, dim=-1)
    txt_simi = torch.matmul(cache_keys_map.reshape(samp_num, -1, feat_dim), clip_weights.t()[cache_labels][:,None,:].permute(0, 2, 1)).squeeze()
    txt_simi, txt_simi_idx = torch.topk(txt_simi, k=30, dim=-1)

    fea_map_img = cache_keys_map.reshape(samp_num, -1, feat_dim)[torch.arange(samp_num)[:, None], img_simi_idx]
    fea_map_txt = cache_keys_map.reshape(samp_num, -1, feat_dim)[torch.arange(samp_num)[:, None], txt_simi_idx]

    fea_map_img = torch.cat((img_simi[:,:, None] * fea_map_img, 10 * cache_keys.t()[:,None,:]), dim=1)
    fea_map_txt = (txt_simi[:,:, None] * fea_map_txt).reshape(cate_num, -1, feat_dim)
    fea_map_txt = torch.cat((fea_map_txt, 10 * clip_weights.t()[:,None,:]), dim=1)

    _, _, Vi = torch.linalg.svd(fea_map_img.reshape(cate_num, -1, feat_dim).mean(dim=1))
    Pimg = torch.matmul(Vi.permute(1,0)[:,:950], Vi[:950:,:])

    _, _, Vt = torch.linalg.svd(fea_map_txt)
    Ptxt = torch.matmul(Vt.permute(0,2,1)[:,:,:950], Vt[:,:950,:])

    proj_clip_weights = torch.matmul(clip_weights.t()[:,None,:], Ptxt).squeeze()
    proj_cache_keys = torch.matmul(cache_keys.t(), Pimg).reshape(-1, feat_dim).t()

    del fea_map_img
    del fea_map_txt
    del cache_keys_map

    proj_test_fea = torch.einsum('ijk,kt',[Ptxt.cpu(), test_features.t().cpu()])
    # proj_test_fea = torch.einsum('ijk,kt', [Ptxt, test_features.t()])
    proj_test_norm = proj_test_fea.norm(dim=1)
    _, proj_test_idx = torch.topk(proj_test_norm, k=10, dim=0)
    txt_test_features = proj_test_fea.permute(2, 0, 1)[torch.arange(test_features.shape[0])[:,None], proj_test_idx.t()].mean(dim=1)

    Ptxt = Ptxt.to(cfg['cuda'])
    test_features = test_features.to(cfg['cuda'])
    txt_test_features = txt_test_features.to(cfg['cuda'])

    img_test_features = test_features @ Pimg

    if cfg['dataset'] == 'imagenet':
        txt_val_features = txt_test_features
        img_val_features = img_test_features
    # else:
    #     proj_val_fea = torch.einsum('ijk,kt',[Ptxt, val_features.t()])
    #     proj_val_norm = proj_val_fea.norm(dim=1)
    #     _, proj_val_idx = torch.topk(proj_val_norm, k=3, dim=0)
    #     txt_val_features = proj_val_fea.permute(2, 0, 1)[torch.arange(val_features.shape[0])[:,None], proj_val_idx.t()].mean(dim=1)        
    #     img_val_features = val_features @ Pimg

    proj_clip_weights = proj_clip_weights / proj_clip_weights.norm(dim=1, keepdim=True)
    proj_cache_keys = proj_cache_keys / proj_cache_keys.norm(dim=0, keepdim=True)

    txt_test_features = txt_test_features / txt_test_features.norm(dim=-1, keepdim=True)
    img_test_features = img_test_features /img_test_features.norm(dim=-1, keepdim=True)

    txt_val_features = txt_val_features / txt_val_features.norm(dim=-1, keepdim=True)
    img_val_features = img_val_features / img_val_features.norm(dim=-1, keepdim=True)

    print(test_labels.shape)
    print(test_features.shape)
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).to(cfg['cuda'])
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).to(cfg['cuda'])
    
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    
    cfg['w'] = cfg['w_training_free']
    # indices = cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False)
    indices = cal_criterion(cfg, proj_clip_weights.t(), proj_cache_keys.t(), only_use_txt=False)
    
    new_proj_clip_weights = proj_clip_weights[:,indices]
    new_proj_cache_keys = proj_cache_keys[indices,:]
    new_txt_test_features =  txt_test_features[:,indices]
    new_img_test_features = img_test_features[:,indices] 

    new_proj_clip_weights = new_proj_clip_weights / new_proj_clip_weights.norm(dim=-1, keepdim=True)
    new_proj_cache_keys = new_proj_cache_keys / new_proj_cache_keys.norm(dim=0, keepdim=True)
    new_img_test_features = new_img_test_features / new_img_test_features.norm(dim=-1, keepdim=True)
    new_txt_test_features = new_txt_test_features / new_txt_test_features.norm(dim=-1, keepdim=True)


    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]
    # new_clip_weights = clip_weights
    # new_cache_keys = cache_keys
    # new_test_features = test_features
    # new_val_features = val_features
    
    new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys /  new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features /  new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features /  new_val_features.norm(dim=-1, keepdim=True)
    
    # # Zero-shot CLIP
    R_fW = 100. * test_features @ clip_weights
    acc = cls_acc(R_fW, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
    
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    
    # Calculate R_f'F'
    # R_fF = new_test_features @ new_cache_keys.t()
    R_fF = new_img_test_features @ new_proj_cache_keys
    
    # Calculate R_F'W'
    # key_logits = new_cache_keys @ new_clip_weights
    key_logits = new_proj_cache_keys.t() @ new_proj_clip_weights.t()
    key_logits = key_logits.softmax(1)
    cache_div = torch.sum(cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
    R_FW = (cache_div * gamma).exp()
    soft_cache_values = cache_values * R_FW
    
    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values
    
    R_fW = 100. * txt_test_features @ proj_clip_weights.t()
    ape_logits = R_fW + cache_logits * alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** Before search, test accuracy: {:.2f}. ****\n".format(acc))
    
    best_search_acc = 0
    # R_fF = new_test_features @ new_cache_keys.t()
    R_fF = new_txt_test_features @ new_proj_cache_keys
    R_fW = 100. * test_features @ clip_weights
    # R_fW = 100. * txt_test_features @ proj_clip_weights.t()

    best_beta, best_alpha, best_gamma = 0, 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    gamma_list = [i * cfg['search_scale'][2] / cfg['search_step'][2] for i in range(cfg['search_step'][2])]
    for beta in beta_list:
        for alpha in alpha_list:
            for gamma in gamma_list:
                with torch.no_grad():
                    soft_cache_values = cache_values * (cache_div * gamma).exp()                    
                    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values
                    ape_logits = R_fW + cache_logits * alpha
                acc = cls_acc(ape_logits, test_labels)
                if acc > best_search_acc:
                    print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(alpha, beta, gamma, acc))
                    best_search_acc = acc
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_search_acc))
    
    R_fW = 100. * test_features @ clip_weights
    # R_fW = 100. * txt_test_features @ proj_clip_weights.t()
    # R_fF = new_test_features @ new_cache_keys.t()
    R_fF = new_txt_test_features @ new_proj_cache_keys

    soft_cache_values = cache_values * (cache_div * best_gamma).exp()
    cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ soft_cache_values
    
    ape_logits = R_fW + cache_logits * best_alpha
    acc = cls_acc(ape_logits, test_labels)
    print("**** APE's test accuracy: {:.2f}. ****\n".format(acc))


def APE_T(cfg, cache_keys, cache_keys_map, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    cache_keys = cache_keys.float().to(cfg['cuda'])
    cache_keys_map = cache_keys_map.float().to(cfg['cuda'])
    cache_values = cache_values.float().to(cfg['cuda'])
    val_features = val_features.float().to(cfg['cuda'])
    val_labels = val_labels.float().to(cfg['cuda'])
    test_features = test_features.float().to(cfg['cuda'])
    test_labels = test_labels.float().to(cfg['cuda'])
    clip_weights = clip_weights.float().to(cfg['cuda'])

    samp_num = cache_keys_map.shape[0]
    feat_dim, cate_num = clip_weights.shape
    cache_labels = torch.argmax(cache_values, -1)
    
    img_simi = torch.matmul(cache_keys_map.reshape(samp_num, -1, feat_dim), cache_keys.t()[:,None,:].permute(0,2,1)).squeeze()
    img_simi, img_simi_idx = torch.topk(img_simi, k=30, dim=-1)
    txt_simi = torch.matmul(cache_keys_map.reshape(samp_num, -1, feat_dim), clip_weights.t()[cache_labels][:,None,:].permute(0, 2, 1)).squeeze()
    txt_simi, txt_simi_idx = torch.topk(txt_simi, k=30, dim=-1)

    fea_map_img = cache_keys_map.reshape(samp_num, -1, feat_dim)[torch.arange(samp_num)[:, None], img_simi_idx]
    fea_map_txt = cache_keys_map.reshape(samp_num, -1, feat_dim)[torch.arange(samp_num)[:, None], txt_simi_idx]

    fea_map_img = torch.cat((img_simi[:,:, None] * fea_map_img, 10 * cache_keys.t()[:,None,:]), dim=1)
    fea_map_txt = (txt_simi[:,:, None] * fea_map_txt).reshape(cate_num, -1, feat_dim)
    fea_map_txt = torch.cat((fea_map_txt, 10 * clip_weights.t()[:,None,:]), dim=1)

    _, _, Vi = torch.linalg.svd(fea_map_img.reshape(cate_num, -1, feat_dim).mean(dim=1))
    Pimg = torch.matmul(Vi.permute(1,0)[:,:950], Vi[:950:,:])

    _, _, Vt = torch.linalg.svd(fea_map_txt)
    Ptxt = torch.matmul(Vt.permute(0,2,1)[:,:,:950], Vt[:,:950,:])

    proj_clip_weights = torch.matmul(clip_weights.t()[:,None,:], Ptxt).squeeze()
    proj_cache_keys = torch.matmul(cache_keys.t(), Pimg).reshape(-1, feat_dim).t()

    del fea_map_img
    del fea_map_txt
    del cache_keys_map

    proj_test_fea = torch.einsum('ijk,kt',[Ptxt.cpu(), test_features.t().cpu()])
    # proj_test_fea = torch.einsum('ijk,kt', [Ptxt, test_features.t()])
    proj_test_norm = proj_test_fea.norm(dim=1)
    _, proj_test_idx = torch.topk(proj_test_norm, k=5, dim=0)
    txt_test_features = proj_test_fea.permute(2, 0, 1)[torch.arange(test_features.shape[0])[:,None], proj_test_idx.t()].mean(dim=1)

    Ptxt = Ptxt.to(cfg['cuda'])
    test_features = test_features.to(cfg['cuda'])
    txt_test_features = txt_test_features.to(cfg['cuda'])

    img_test_features = test_features @ Pimg

    if cfg['dataset'] == 'imagenet':
        txt_val_features = txt_test_features
        img_val_features = img_test_features
    # else:
    #     proj_val_fea = torch.einsum('ijk,kt',[Ptxt, val_features.t()])
    #     proj_val_norm = proj_val_fea.norm(dim=1)
    #     _, proj_val_idx = torch.topk(proj_val_norm, k=3, dim=0)
    #     txt_val_features = proj_val_fea.permute(2, 0, 1)[torch.arange(val_features.shape[0])[:,None], proj_val_idx.t()].mean(dim=1)        
    #     img_val_features = val_features @ Pimg

    proj_clip_weights = proj_clip_weights / proj_clip_weights.norm(dim=1, keepdim=True)
    proj_cache_keys = proj_cache_keys / proj_cache_keys.norm(dim=0, keepdim=True)

    txt_test_features = txt_test_features / txt_test_features.norm(dim=-1, keepdim=True)
    img_test_features = img_test_features /img_test_features.norm(dim=-1, keepdim=True)

    txt_val_features = txt_val_features / txt_val_features.norm(dim=-1, keepdim=True)
    img_val_features = img_val_features / img_val_features.norm(dim=-1, keepdim=True)


    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim)
    
    cfg['w'] = cfg['w_training']
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    # adapter = APE_Training(cfg, clip_weights, clip_model, cache_keys).cuda()
    adapter = APE_Training_F(cfg, clip_weights, clip_model, cache_keys).to(cfg['cuda'])
    # adapter = APE_Training_F(cfg, proj_clip_weights.t(), clip_model, proj_cache_keys.t()).to(cfg['cuda'])

    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=cfg['eps'], weight_decay=1e-1)  # 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    Loss = SmoothCrossEntropy()
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    # feat_num = cfg['feat_num']
    
    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(cfg['cuda']), target.to(cfg['cuda'])
            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

                vis_image_features = image_features @ Pimg
                vis_image_features /= vis_image_features.norm(dim=-1, keepdim=True)

                proj_fea = torch.einsum('ijk,kt', Ptxt, image_features.t())
                proj_fea_norm = proj_fea.norm(dim=1)
                _, proj_idx = torch.topk(proj_fea_norm, k=5, dim=0)
                txt_image_features = proj_fea.permute(2, 0, 1)[torch.arange(image_features.shape[0])[:,None], proj_idx.t()].mean(dim=1) 
                txt_image_features /= txt_image_features.norm(dim=1, keepdim=True)

            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
            R_fF = image_features @ new_cache_keys.t()
            # R_fF = vis_image_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * image_features @ new_clip_weights
            # R_fW = 100. * txt_image_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha

            loss = Loss(ape_logits, target)

            acc = cls_acc(ape_logits, target)
            correct_samples += acc / 100 * len(ape_logits)
            all_samples += len(ape_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        with torch.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)

            R_fF = val_features @ new_cache_keys.t()
            # R_fF = img_val_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * val_features @ new_clip_weights
            # R_fW = 100. * txt_val_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha
        acc = cls_acc(ape_logits, val_labels)

        print("**** APE-T's test accuracy: {:.2f}. ****\n".format(acc))
        if acc >= best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter, cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt")
    
    adapter = torch.load(cfg['cache_dir'] + "/APE-T_" + str(cfg['shots']) + "shots.pt").to(cfg['cuda'])
    print(f"**** After fine-tuning, APE-T's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    best_search_acc = 0
    best_beta, best_alpha = 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    for beta in beta_list:
        for alpha in alpha_list:
            with torch.no_grad():
                new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
                
                R_fF = val_features @ new_cache_keys.t()
                # R_fF = img_test_features @ new_cache_keys.t()
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
                R_fW = 100. * val_features @ new_clip_weights
                # R_fW = 100. * txt_test_features @ new_clip_weights
                ape_logits = R_fW + cache_logits * alpha
            acc = cls_acc(ape_logits, test_labels)
            if acc > best_search_acc:
                print("New best setting, alpha: {:.2f}, beta: {:.2f}; accuracy: {:.2f}".format(alpha, beta, acc)) 
                best_search_acc = acc
                best_alpha, best_beta = alpha, beta
    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_search_acc))
    
    # print("\n-------- Evaluating on the test set. --------")
    # with torch.no_grad():
    #     new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
        
    #     # R_fF = test_features @ new_cache_keys.t()
    #     R_fF = img_test_features @ new_cache_keys.t()
    #     cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ R_FW
    #     R_fW = 100. * test_features @ new_clip_weights
    #     # R_fW = 100. * txt_test_features @ new_clip_weights
    #     ape_logits = R_fW + cache_logits * best_alpha
    # acc = cls_acc(ape_logits, test_labels)
    # print("**** APE-T's test accuracy: {:.2f}. ****\n".format(acc))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    print(cfg['shots'])

    # cache_dir =  os.path.join('./caches_orig', cfg['dataset'])
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    cfg['cuda'] = torch.device("cuda:1")

    print(cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.to(cfg['cuda'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = load_text_feature(cfg)
    clip_weights = clip_weights.to(cfg['cuda'])

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)
    cache_keys_map = load_few_shot_feature_map(cfg)
    cache_keys = cache_keys.to(cfg['cuda'])
    cache_values = cache_values.to(cfg['cuda'])
    cache_keys_map = cache_keys_map.to(cfg['cuda'])


    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    val_features = val_features.to(cfg['cuda'])
    val_labels = val_labels.to(cfg['cuda'])

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg['dataset'] == 'imagenet':
        # test_features, test_labels = loda_val_test_feature(cfg, "val")
        test_features, test_labels = val_features, val_labels
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")

    test_features = test_features.to(cfg['cuda'])
    test_labels = test_labels.to(cfg['cuda'])

    # # # ------------------------------------------  APE  ------------------------------------------
    APE(cfg, cache_keys, cache_keys_map, cache_values, test_features, test_labels,  test_features, test_labels, clip_weights)

    # # ------------------------------------------ APE-T ------------------------------------------
    if cfg['dataset'] == 'imagenet':
        imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)
    else:   
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Resize(size=224, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=512, tfm=train_tranform, is_train=True, shuffle=True)
    APE_T(cfg, cache_keys, cache_keys_map, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)
    
    

if __name__ == '__main__':
    main()
