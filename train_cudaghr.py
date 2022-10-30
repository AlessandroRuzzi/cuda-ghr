import numpy as np
from collections import OrderedDict
import json
import argparse
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import losses
from dataset import HDFDataset
from nets import Controller
import torch.nn.functional as F
from utils import adjust_learning_rate, RunningStatistics, send_data_dict_to_gpu, recover_images, build_image_matrix
from argparse import Namespace
import warnings
import random
from torchvision import transforms
import wandb
from PIL import Image
from xgaze_dataloader import get_train_loader
from xgaze_dataloader import get_val_loader as xgaze_get_val_loader
from standard_image_dataset import get_data_loader as image_get_data_loader
from mpii_face_dataloader import get_val_loader as mpii_get_val_loader
from columbia_dataloader import get_val_loader as columbia_get_val_loader
from gaze_capture_dataloader import get_val_loader as gaze_capture_get_val_loader
from piq import ssim, psnr, LPIPS, FID
from gaze_estimation_utils import normalize
import scipy.io
from logging_utils import log_evaluation_image, log_one_subject_evaluation_results, log_all_datasets_evaluation_results
from face_recognition.evaluation_similarity import evaluation_similarity
from nets.xgaze_baseline_head import gaze_network_head

warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
    ])

trans_normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  
    ])

trans_resize = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224, 224)),
    ]
)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        

torch.manual_seed(45)  # cpu
torch.cuda.manual_seed(55)  # gpu
np.random.seed(65)  # numpy
random.seed(75)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

wandb.init(project="cuda-ghr", config={"gpu_id": 0})

parser = argparse.ArgumentParser(description='Train a CUDA-GHR controller.')
parser.add_argument('--config_json', type=str, default= "configs/config_xgaze_to_xgaze.json",  help='Path to config in JSON format')
parser.add_argument('--columbia', default=False, help='train on columbia if true')
args = parser.parse_args()

#####################################################
# load configurations
assert os.path.isfile(args.config_json)
logging.info('Loading ' + args.config_json)
config = json.load(open(args.config_json))
config = Namespace(**config)
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

config.lr = config.batch_size*config.base_learning_rate

#####################################################
# load datasets
source_train_dataset, source_train_dataloader = get_train_loader(data_dir = "/data/aruzzi/xgaze_subjects",batch_size=int(config.batch_size), key_data='source')
target_train_dataset, target_train_dataloader = get_train_loader(data_dir = "/data/aruzzi/xgaze_subjects",batch_size=int(config.batch_size), key_data= 'target')

# logging data stats.
logging.info('')
logging.info("Source datset size: %s" % len(source_train_dataset))
logging.info("Target dataset size: %s " % len(target_train_dataset))

#####################################################
# create network

network = Controller(config, device)
network = network.to(device)

if torch.cuda.device_count() >= 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network.push_modules_to_multi_gpu()


if config.load_step != 0:
    logging.info('Loading available model')
    network.load_model(os.path.join(config.save_path, "checkpoints", str(config.load_step) + '.pt'))

if 'pretrained' in config:
    logging.info('Loading pretrained model')
    network.load_model(config.pretrained)


gen_params = list(network.image_encoder.parameters()) + \
             list(network.generator.parameters()) + \
             list(network.gaze_latent_encoder.parameters())

gen_optimizer = torch.optim.Adam(gen_params, lr=config.lr, weight_decay=config.l2_reg)

disc_params = list(network.target_discrim.parameters()) + \
              list(network.source_discrim.parameters()) + \
              list(network.latent_discriminator.parameters())

disc_optimizer = torch.optim.Adam(disc_params, lr=config.lr * 0.001, weight_decay=config.l2_reg)

optimizers = [gen_optimizer, disc_optimizer]

perceptual_loss = losses.PerceptualLoss(device=device)

latent_disc_loss = nn.BCEWithLogitsLoss(reduction='mean')

#####################################################

# single training step
def execute_training_step(current_step):

    network.train_mode_on()

    global source_train_data_iterator
    global target_train_data_iterator
    try:
        source_input = next(source_train_data_iterator)
        target_input = next(target_train_data_iterator)
    except StopIteration:
        torch.cuda.empty_cache()
        global source_train_dataloader
        global target_train_dataloader
        source_train_data_iterator = iter(source_train_dataloader)
        target_train_data_iterator = iter(target_train_dataloader)
        source_input = next(source_train_data_iterator)
        target_input = next(target_train_data_iterator)

    input_dict = send_data_dict_to_gpu(source_input, target_input, device)

    ############### DISCRIMINATOR ############

    for param in network.target_discrim.parameters():
        param.requires_grad = True
    for param in network.source_discrim.parameters():
        param.requires_grad = True
    for param in network.latent_discriminator.parameters():
        param.requires_grad = True
    for param in network.image_encoder.parameters():
        param.requires_grad = False
    for param in network.generator.parameters():
        param.requires_grad = False
    for param in network.gaze_latent_encoder.parameters():
        param.requires_grad = False

    ## source data
    s_app_embedding = network.image_encoder(input_dict['source_image'])
    s_gaze_embedding = network.gaze_latent_encoder(input_dict['source_gaze'])
    source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
    source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                                  mode='constant', value=0)])[0]

    real = network.source_discrim(input_dict['source_image'])
    fake = network.source_discrim(source_gen_image.detach())
    disc_loss_D = losses.discriminator_loss(real=real, fake=fake)

    ## target data
    gaze_pred_, head_pred_ = network.task_net((input_dict['target_image']-mean)/std)
    t_app_embedding = network.image_encoder(input_dict['target_image'])
    t_gaze_embedding = network.gaze_latent_encoder(gaze_pred_)
    target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
    target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_, pad=(0, 1),
                                                           mode='constant', value=0)])[0]

    real = network.target_discrim(input_dict['target_image'])
    fake = network.target_discrim(target_gen_image.detach())
    disc_loss_D += losses.discriminator_loss(real=real, fake=fake)

    ## latents
    real = network.latent_discriminator(target_embedding)
    fake = network.latent_discriminator(source_emdedding)
    disc_loss_D += losses.discriminator_loss(real=real, fake=fake)

    disc_optimizer.zero_grad()
    disc_loss_D.backward()
    disc_optimizer.step()

    ############### GENERATOR ############

    for param in network.target_discrim.parameters():
        param.requires_grad = False
    for param in network.source_discrim.parameters():
        param.requires_grad = False
    for param in network.latent_discriminator.parameters():
        param.requires_grad = False
    for param in network.image_encoder.parameters():
        param.requires_grad = True
    for param in network.generator.parameters():
        param.requires_grad = True
    for param in network.gaze_latent_encoder.parameters():
        param.requires_grad = True

    # source data
    s_app_embedding = network.image_encoder(input_dict['source_image'])
    s_gaze_embedding = network.gaze_latent_encoder(input_dict['source_gaze'])
    source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
    source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                                  mode='constant', value=0)])[0]
    # target data
    gaze_pred_orig_t, head_pred_orig_t = network.task_net((input_dict['target_image']-mean)/std)
    t_app_embedding = network.image_encoder(input_dict['target_image'])
    t_gaze_embedding = network.gaze_latent_encoder(gaze_pred_orig_t)
    target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
    target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_orig_t, pad=(0, 1),
                                                           mode='constant', value=0)])[0]

    ## reconstruction loss and perceptual loss
    rloss = torch.nn.L1Loss()(source_gen_image, input_dict['source_image']) + \
            torch.nn.L1Loss()(target_gen_image, input_dict['target_image'])

    percep_loss = perceptual_loss.loss(source_gen_image, input_dict['source_image']) \
                  + perceptual_loss.loss(target_gen_image, input_dict['target_image'])

    ## GAN loss
    fake = network.source_discrim(source_gen_image)
    disc_loss_G = losses.generator_loss(fake=fake)

    fake = network.target_discrim(target_gen_image)
    disc_loss_G += losses.generator_loss(fake=fake)

    # L_feat
    fake = network.latent_discriminator(target_embedding)
    real = network.latent_discriminator(source_emdedding)
    real_size = list(real.size())
    fake_size = list(fake.size())
    real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
    fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)
    disc_loss_G += (latent_disc_loss(fake, fake_label) + latent_disc_loss(real, real_label)) / 2

    ## label consistency loss
    gaze_pred_orig_s, head_pred_orig_s = network.task_net((input_dict['source_image']-mean)/std)
    gaze_pred, head_pred = network.task_net((source_gen_image-mean)/std)
    task_loss = losses.gaze_angular_loss(y=gaze_pred_orig_s, y_hat=gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_s, y_hat=head_pred)

    gaze_pred, head_pred = network.task_net((target_gen_image-mean)/std)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_t, y_hat=gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_t, y_hat=head_pred)

    ## redirected consistency loss
    gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                            F.pad(input=head_pred_orig_t, pad=(0, 1),
                                                  mode='constant', value=0)])[0]
    head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                            F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                  mode='constant', value=0)])[0]

    gaze_swapped_gaze_pred, gaze_swapped_head_pred = network.task_net((gaze_swapped_image-mean)/std)
    head_swapped_gaze_pred, head_swapped_head_pred = network.task_net((head_swapped_image-mean)/std)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_s, y_hat=gaze_swapped_gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_t, y_hat=gaze_swapped_head_pred)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_t, y_hat=head_swapped_gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_s, y_hat=head_swapped_head_pred)

    loss = config.coeff_l1_loss*rloss + \
           config.coeff_latent_discriminator_loss*disc_loss_G + \
           config.coeff_perc_loss*percep_loss + \
           config.coeff_gaze_loss*task_loss

    gen_optimizer.zero_grad()
    loss.backward()
    gen_optimizer.step()


    if current_step % 50 == 0:
        img = np.concatenate([((input_dict['source_image'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8),((input_dict['target_image'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8),np.clip(((target_gen_image.detach().cpu().permute(0, 2, 3, 1).numpy()  +1) * 255.0/2.0),0,255).astype(np.uint8)],axis=2)
        #img = np.concatenate([(input['image_a'].detach().cpu().permute(0, 2, 3, 1).numpy()* 255.0).astype(np.uint8),(input['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8),(generated.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)],axis=2)
        img = Image.fromarray(img[0])
        log_image = wandb.Image(img)
        #log_image.show()
        wandb.log({"Sted Prediction": log_image})

    # save training samples in tensorboard
    if config.use_tensorboard and current_step % config.save_freq_images == 0 and current_step != 0:
        image_index = 0
        tensorboard.add_image('train/source_input_image',
                              torch.clamp((input_dict['source_image'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_input_image',
                              torch.clamp((input_dict['target_image'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/source_generated_image',
                              torch.clamp((source_gen_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_image',
                              torch.clamp((target_gen_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_gaze_swap_image',
                              torch.clamp((gaze_swapped_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_head_swap_image',
                              torch.clamp((head_swapped_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)

    return rloss.item(), disc_loss_D.item(), disc_loss_G.item(), percep_loss.item(), task_loss.item()

def select_dataloader(name, subject, idx, img_dir, batch_size, num_images, num_workers, is_shuffle):
    if name == "eth_xgaze":
        return (name, subject, idx, xgaze_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "mpii_face_gaze":
        return (name, subject, idx, mpii_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "columbia":
        return (name, subject, idx, columbia_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "gaze_capture":
        return (name, subject, idx, gaze_capture_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    else:
        print("Dataset not supported")

def select_cam_matrix(name,cam_matrix,cam_distortion,cam_ind, subject):
    if name == "eth_xgaze":
        return cam_matrix[name][cam_ind], cam_distortion[name][cam_ind]
    elif name == "mpii_face_gaze":
        camera_matrix = cam_matrix[name][int(subject[-5:-3])]
        camera_matrix[0, 2] = 256.0
        camera_matrix[1, 2] = 256.0
        return camera_matrix, cam_distortion[name][int(subject[-5:-3])]
    elif name == "columbia":
        return cam_matrix[name], cam_distortion[name]
    elif name == "gaze_capture":
        pass
    else:
        print("Dataset not supported")

def load_cams():
    cam_matrix = {}
    cam_distortion = {}
    cam_translation = {}
    cam_rotation = {}

    for name in config.data_names:
        cam_matrix[name] = []
        cam_distortion[name] = []
        cam_translation[name] = []
        cam_rotation[name] = []
    

    for cam_id in range(18):
        cam_file_name = "data/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
        fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
        cam_matrix["eth_xgaze"].append(fs.getNode("Camera_Matrix").mat())
        cam_distortion["eth_xgaze"].append(fs.getNode("Distortion_Coefficients").mat())
        cam_translation["eth_xgaze"].append(fs.getNode("cam_translation"))
        cam_rotation["eth_xgaze"].append(fs.getNode("cam_rotation"))
        fs.release()

    for i in range(15):
        file_name = os.path.join(
        "data/mpii_face_gaze/cam", "Camera" + str(i).zfill(2) + ".mat"
        )
        mat = scipy.io.loadmat(file_name)
        cam_matrix["mpii_face_gaze"].append(mat.get("cameraMatrix"))
        cam_distortion["mpii_face_gaze"].append(mat.get(
            "distCoeffs"
        ))

    cam_file_name = "data/columbia/cam/cam" + str(0).zfill(2) + ".xml"
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    cam_matrix["columbia"] = fs.getNode("Camera_Matrix").mat()
    cam_distortion["columbia"] = fs.getNode("Distortion_Coefficients").mat()

    return cam_matrix,cam_distortion, cam_translation, cam_rotation

def calculate_FID(gt_images, pred_images):
    first_dl, second_dl = image_get_data_loader(gt_images), image_get_data_loader(pred_images)
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(first_dl)
    second_feats = fid_metric.compute_feats(second_dl)
    fid: torch.Tensor = fid_metric(first_feats, second_feats)
    return fid

def execute_test(log, current_step):

    face_model_load =  np.loadtxt('data/eth_xgaze/face_model.txt')  # Generic face model with 3D facial landmarks
    val_keys = {}
    for name in config.data_names:
        file_path = os.path.join("data", name, "train_test_split.json")
        with open(file_path, "r") as f:
            datastore = json.load(f)
        val_keys[name] = datastore["val"]

    dataloader_all = []

    for idx,name in enumerate(config.data_names):
        for subject in val_keys[name]:
            dataloader_all.append(select_dataloader(name, subject, idx, config.img_dir[idx], 1, config.num_images[idx], 0, is_shuffle=False))   

    cam_matrix, cam_distortion, cam_translation, cam_rotation = load_cams()


    path = "sted/checkpoints/epoch_24_resnet_80_head_ckpt.pth.tar"
    model = gaze_network_head().to(device)

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict=state_dict["model_state"])
    model.eval()

    print("Done")

    dict_angular_loss = {}
    dict_angular_head_loss = {}
    dict_ssim_loss = {}
    dict_psnr_loss = {}
    dict_lpips_loss = {}
    dict_l1_loss = {}
    dict_num_images = {}

    dict_similarity = {}

    dict_fid = {}
    dict_gt_images = {}
    dict_pred_images = {}
    full_images_gt_list = []
    full_images_pred_list = []

    for name in config.data_names:
        dict_angular_loss[name] = 0.0
        dict_angular_head_loss[name] = 0.0
        dict_ssim_loss[name] = 0.0
        dict_psnr_loss[name] = 0.0
        dict_lpips_loss[name] = 0.0
        dict_l1_loss[name] = 0.0
        dict_num_images[name] = 0

        dict_similarity[name] = 0.0

        dict_fid[name] = 0.0
        dict_gt_images[name] = []
        dict_pred_images[name] = []
    
    for name, subject, index_dataset, dataloader in dataloader_all:
    
        angular_loss = 0.0
        angular_head_loss = 0.0
        ssim_loss = 0.0
        psnr_loss = 0.0
        lpips_loss = 0.0
        l1_loss = 0.0
        num_images = 0

        similarity = 0.0

        fid = 0.0
        gt_list = []
        pred_list = []

        for index,entry in enumerate(dataloader):
            print(index)
            ldms = entry["ldms_b"][0]
            batch_head_mask = torch.reshape(entry["mask_b"], (1, 1, 512, 512))
            cam_ind = entry["cam_ind_b"]

            camera_matrix, camera_distortion = select_cam_matrix(name, cam_matrix,cam_distortion, cam_ind, subject)

            input_dict = send_data_dict_to_gpu(entry, device)
            s_app_embedding = network.image_encoder(input_dict['source_image'])
            s_gaze_embedding = network.gaze_latent_encoder(input_dict['gaze_b'])
            source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
            source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['head_b'], pad=(0, 1),
                                                                  mode='constant', value=0)])[0]

            image_gt = ((input_dict['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8)
            image_gen = np.clip(((source_gen_image.detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0),0,255).astype(np.uint8)

            batch_images_gt = trans_normalize(image_gt[0,:])
            nonhead_mask = batch_head_mask < 0.5   
            nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)  
            batch_images_gt = torch.reshape(batch_images_gt,(1,3,512,512))      
            batch_images_gt[nonhead_mask_c3b] = 1.0

            target_image_quality = torch.reshape(
                batch_images_gt , (1, 3, 512, 512)
            ).to(device)

            batch_images_gt_norm = normalize(
                (batch_images_gt.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                    np.uint8
                )[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                config.img_dim,
            )
            target_normalized_log = torch.reshape(trans_normalize(batch_images_gt_norm),(1,3,224,224)).to(device)
            batch_images_gt_norm = torch.reshape(
                trans(batch_images_gt_norm), (1, 3, config.img_dim, config.img_dim)
            ).to(
                device
            )  
            pitchyaw_gt, head_gt = model(batch_images_gt_norm)

            batch_images_gen = trans_normalize(image_gen[0,:])
            nonhead_mask = batch_head_mask < 0.5
            nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
            batch_images_gen = torch.reshape(batch_images_gen,(1,3,512,512))
            batch_images_gen[nonhead_mask_c3b] = 1.0

            pred_image_quality = torch.reshape(
                 batch_images_gen, (1, 3, 512, 512)
            ).to(device)

            batch_images_gen_norm = normalize(
                (batch_images_gen.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                    np.uint8
                )[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                config.img_dim,
            )

            pred_normalized_log = torch.reshape(trans_normalize(batch_images_gen_norm),(1,3,224,224)).to(device)
            
            batch_images_norm = torch.reshape(
                trans(batch_images_gen_norm), (1, 3, config.img_dim, config.img_dim)
            ).to(device)
            pitchyaw_gen, head_gen = model(batch_images_norm)


            loss = losses.gaze_angular_loss(pitchyaw_gt,pitchyaw_gen).detach().cpu().numpy()
            angular_loss += loss
            num_images += 1
            dict_angular_loss[name] += loss
            dict_num_images[name] += 1
            print("Gaze Angular Error: ",angular_loss/num_images,loss,num_images)

            loss = losses.gaze_angular_loss(head_gt,head_gen).detach().cpu().numpy()
            angular_head_loss += loss
            dict_angular_head_loss[name] += loss
            print("Head Angular Error: ",angular_head_loss/num_images,loss,num_images)

            sim_gt = ( torch.reshape(
                trans_resize(batch_images_gt[0,:]) , (1, 3, config.img_dim, config.img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]

            sim_gen = ( torch.reshape(
                trans_resize(batch_images_gen[0,:]) , (1, 3, config.img_dim, config.img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]
            try:
                loss = evaluation_similarity(sim_gt, sim_gen)
            except:
                loss = -0.1
            similarity += loss
            dict_similarity[name] += loss
            print("Similarity Score: ", similarity / num_images, loss, num_images)

            gt_list.append(target_image_quality[0,:])
            pred_list.append(pred_image_quality[0,:])

            dict_gt_images[name].append(target_image_quality[0,:])
            dict_pred_images[name].append(pred_image_quality[0,:])

            full_images_gt_list.append(target_image_quality[0,:])
            full_images_pred_list.append(pred_image_quality[0,:])

            loss = ssim(target_image_quality, pred_image_quality, data_range=1.).detach().cpu().numpy()
            ssim_loss += loss
            dict_ssim_loss[name] += loss
            print("SSIM: ",ssim_loss/num_images,loss,num_images)

            loss = psnr(target_image_quality, pred_image_quality, data_range=1.).detach().cpu().numpy()
            psnr_loss += loss
            dict_psnr_loss[name] += loss
            print("PSNR: ",psnr_loss/num_images,loss,num_images)

            lpips_metric = LPIPS()
            loss = lpips_metric(target_image_quality, pred_image_quality).detach().cpu().numpy()
            lpips_loss += loss
            dict_lpips_loss[name] += loss
            print("LPIPS: ",lpips_loss/num_images,loss,num_images)

            loss = torch.nn.functional.l1_loss(target_image_quality, pred_image_quality).detach().cpu().numpy()
            l1_loss += loss
            dict_l1_loss[name] += loss
            print("L1 Distance: ", l1_loss/num_images,loss, num_images)

            if index % log == 0:
                log_evaluation_image(pred_normalized_log, target_normalized_log, ((input_dict['source_image'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8), image_gt, image_gen)

        fid = calculate_FID(gt_images= gt_list, pred_images= pred_list)

        if index % log == 0:
            log_one_subject_evaluation_results(current_step, angular_loss, angular_head_loss, ssim_loss, psnr_loss, lpips_loss,
                                                l1_loss, num_images, fid , similarity)

    for name in config.data_names:
        dict_fid[name]  = calculate_FID(gt_images= dict_gt_images[name], pred_images= dict_pred_images[name])
        
    full_fid = calculate_FID(gt_images= full_images_gt_list, pred_images= full_images_pred_list)    
                            
    if index % log == 0:
        log_all_datasets_evaluation_results(current_step, config.data_names, dict_angular_loss, dict_angular_head_loss, dict_ssim_loss, dict_psnr_loss, dict_lpips_loss,
                                                dict_l1_loss, dict_num_images,dict_fid, full_fid, dict_similarity)



#####################################################
# initializing tensorboard

if config.use_tensorboard:
    from tensorboardX import SummaryWriter
    tensorboard = SummaryWriter(logdir=config.save_path)

#####################################################

logging.info('Training')
running_losses = RunningStatistics()
source_train_data_iterator = iter(source_train_dataloader)
target_train_data_iterator = iter(target_train_dataloader)

# fixing test samples
#target_test_data_dict = next(target_train_data_iterator)
#source_test_data_dict = next(source_train_data_iterator)

#test_data_dict = send_data_dict_to_gpu(source_test_data_dict, target_test_data_dict, device)

if config.train_mode:

    #####################################################
    # save configurations only if training
    target_dir = config.save_path + '/configs'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    fpath = os.path.relpath(target_dir + '/params.json')
    with open(fpath, 'w') as f:
        json.dump(vars(config), f, indent=4)
        logging.info('Written %s' % fpath)

    #####################################################

    # main training loop
    for current_step in range(config.load_step, config.num_training_steps):
        # Save model
        if current_step % config.save_interval == 0 and current_step != config.load_step:
            network.save_model(current_step)

        # lr decay
        if (current_step % config.decay_steps == 0) or current_step == config.load_step:
            lr = adjust_learning_rate(optimizers, config.decay, int(current_step / config.decay_steps), config.lr)
            if config.use_tensorboard:
                tensorboard.add_scalar('train/lr', lr, current_step)

        # Testing loop: every specified iterations compute the test statistics
        if current_step % config.print_freq_test == 0 and current_step != config.load_step:
            network.eval()
            #network.clean_up()
            torch.cuda.empty_cache()
            execute_test(1, current_step)
            # This might help with memory leaks
            torch.cuda.empty_cache()

        # Training step
        tr_loss = execute_training_step(current_step)
        # Print training loss
        if current_step != 0 and (current_step % config.print_freq_train == 0):
            logging.info('Losses at [%7d]: %s %s %s %s %s' %
                         (current_step, tr_loss[0], tr_loss[1], tr_loss[2], tr_loss[3], tr_loss[4]))
            wandb.log({'train/l1_loss': tr_loss[0]})
            wandb.log({'train/discD': tr_loss[1]})
            wandb.log({'train/discG': tr_loss[2]})
            wandb.log({'train/perc_loss': tr_loss[3]})
            wandb.log({'train/task_loss': tr_loss[4]})
            if config.use_tensorboard:
                tensorboard.add_scalar('train/l1_loss', tr_loss[0], current_step)
                tensorboard.add_scalar('train/discD', tr_loss[1], current_step)
                tensorboard.add_scalar('train/discG', tr_loss[2], current_step)
                tensorboard.add_scalar('train/perc_loss', tr_loss[3], current_step)
                tensorboard.add_scalar('train/task_loss', tr_loss[4], current_step)

    logging.info('Finished Training')

    # save final model
    network.save_model(config.num_training_steps)

#####################################################
