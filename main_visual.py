from utils import visual_network
from dataset import dataset
import torch
import argparse
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import os
import pickle
from fvcore.nn import FlopCountAnalysis
import time
import datetime
import wandb
from prettytable import PrettyTable


# Parser options - 
parser = argparse.ArgumentParser()
parser.add_argument('--action', required=True, type=str, help='Action: [train, test, gzsl_test, fsl_train, fsl_test, sup_train, sup_test]')
parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, olympics, test]')
parser.add_argument('--dataset_path', required=True, type=str, help='Path of the datasets')
parser.add_argument('--split_index', required=True, type=int, help='Index for splitting of classes')
parser.add_argument('--novelty', default='LCA', type=str, help='Novel component: [LCA, other]')
parser.add_argument('--early_stop_thresh', default=150, type=int, help='Number of training epochs before early stopping')
parser.add_argument('--t_blocks', default=1, type=int, help='Number of transformer blocks')


parser.add_argument('--network', default='r2plus1d', type=str, help='Network backend choice: [c3d, r2plus1d, clip, clip_classifier]')
parser.add_argument('--vit_backbone', default='ViT-B/16', type=str, help='Backbonesof clip: [ViT-B/16, ViT-B/32, ViT-L/14, RN50, RN101]')
parser.add_argument('--lca_branch', default=3, type=int, help='Number of LCA dilation branches.')
parser.add_argument('--semantic', default='word2vec', type=str, help='Semantic choice: [word2vec, fasttext, sent2vec, clip, clip_manual]')

parser.add_argument('--clip_len', default=16, type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips_train', default=1, type=int, help='Number of clips per video (training)')
parser.add_argument('--n_clips_test', default=25, type=int, help='Number of clips per video (testing)')
parser.add_argument('--image_size', default=224, type=int, help='Image size in input.')

parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs', default=150, type=int, help='Number of training epochs.')
parser.add_argument('--batch_size', default=22, type=int, help='Mini-Batchsize size per GPU.')
parser.add_argument('--drop_attn_prob', default=0.0, type=float, help='Dropout probability for MHSA module.')
parser.add_argument('--droppath', default=0.0, type=float, help='Drop path probability.')



parser.add_argument('--fixconvs', action='store_false', default=True, help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True, help='Pretrain network.')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers for training.')

parser.add_argument('--trained_weights', default=False, type=str, help='Use trained weights of Brattoli on Kinetics')
parser.add_argument('--no_val', action='store_true', default=False, help='Perform no validation')
parser.add_argument('--val_freq', default=2, type=int, help='Frequency for running validation.')

parser.add_argument('--random_seed', default=806, help='Seed for initialization')
parser.add_argument('--count_params', action='store_true', default=False, help='Only for counting trainable parameters')


# parser.add_argument(
#         "--precision",
#         choices=["amp", "fp16", "fp32"],
#         default="amp",
#         help="Floating point precition."
#     ) 

parser.add_argument('--save_path', required=True, type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights', default=None, type=str, help='Weights to load from a previously run.')
opt = parser.parse_args()

def train(train_dataloader, model, optimizer, criterion, opt, epoch):
    model.train()
    # autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    
    # Perform one epoch of training.
    class_embd = train_dataloader.dataset.class_embd
    accuracies = []
    losses = []

    data_iterator = tqdm(train_dataloader)
    for i, (video, label, embd, _, _) in enumerate(data_iterator):
        
        pred_embd = model(video.to(opt.device))
        embd = embd.to(opt.device)

        pred_embd_np = pred_embd.detach().cpu().numpy()
        # Using cosine distance for predicted label.
        pred_label = cdist(pred_embd_np, class_embd, 'cosine').argmin(axis=1)
        acc = accuracy_score(label.numpy(), pred_label) * 100
        accuracies.append(acc)

        loss = criterion(pred_embd, embd)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    accuracy = np.mean(accuracies)
    loss = np.mean(losses)
    # Logging results.
    with open(opt.results_path, 'a') as f:
        #new change - written loss
        f.write('Epoch = %d, Train Accuracy = %.2f, Train Loss = %.4f\n' % (epoch, accuracy, loss))
    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(accuracy, f)

    # return train loss too
    return accuracy, loss


def compute_accuracy(predicted_embd, class_embd, true_embd):
    # Compute top-1 and top-5 accuracy.
    y_pred = cdist(predicted_embd, class_embd, 'cosine').argsort(1)
    y = cdist(true_embd, class_embd, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100

    classwise_accuracy = [0 for i in range(len(class_embd))]
    classwise_num_test = [0 for i in range(len(class_embd))]
    for i in range(len(y_pred)):
        if y_pred[i][0] == y[i]:
            classwise_accuracy[y[i]]+=1
        classwise_num_test[y[i]]+=1
    classwise_accuracy = [((acc / num) * 100) for (acc, num) in zip(classwise_accuracy, classwise_num_test)]
    return accuracy, accuracy_top5, classwise_accuracy


def compute_generalized_accuracy(predicted_embd, class_embd, true_embd, seens):
    # Compute top-1 and top-5 accuracy.
    y_pred = cdist(predicted_embd, class_embd, 'cosine').argsort(1)
    y = cdist(true_embd, class_embd, 'cosine').argmin(1)

    y_pred_seen = [y_e for (y_e, seen) in zip(y_pred, seens) if seen == True]
    y_pred_unseen = [y_e for (y_e, seen) in zip(y_pred, seens) if seen == False]
    y_seen = [y_e for (y_e, seen) in zip(y, seens) if seen == True]
    y_unseen = [y_e for (y_e, seen) in zip(y, seens) if seen == False]

    correct_seen = np.asarray([1 for ele, ele_pred in zip(y_seen, y_pred_seen) if ele == ele_pred[0]])
    correct_seen = np.sum(correct_seen)
    accuracy_seen = (correct_seen / len(y_seen)) * 100

    correct_unseen = np.asarray([1 for ele, ele_pred in zip(y_unseen, y_pred_unseen) if ele == ele_pred[0]])
    correct_unseen = np.sum(correct_unseen)
    accuracy_unseen = (correct_unseen / len(y_unseen)) * 100

    # accuracy_seen = accuracy_score(y_seen, y_pred_seen[:, 0]) * 100
    # accuracy_unseen = accuracy_score(y_unseen, y_pred_unseen[:, 0]) * 100
    accuracy = (2 * accuracy_seen * accuracy_unseen) / (accuracy_seen + accuracy_unseen)

    classwise_accuracy = [0 for i in range(len(class_embd))]
    classwise_num_test = [0 for i in range(len(class_embd))]
    for i in range(len(y_pred)):
        if y_pred[i][0] == y[i]:
            classwise_accuracy[y[i]]+=1
        classwise_num_test[y[i]]+=1
    classwise_accuracy = [((acc / num) * 100) for (acc, num) in zip(classwise_accuracy, classwise_num_test)]
    return accuracy, accuracy_seen, accuracy_unseen, classwise_accuracy


def test(test_dataloader, model, opt, criterion):
    # Perform testing.
    model.eval()
    val_losses = []
    with torch.no_grad():
        n_samples = len(test_dataloader.dataset)
        
        if semantic == 'word2vec' or semantic == 'fasttext':
            output_features = 300
        elif semantic == 'sent2vec':
            output_features = 600
        elif 'clip' in semantic:
            if opt.vit_backbone in ['ViT-B/16', 'ViT-B/32', 'RN101']:
                output_features = 512
            elif opt.vit_backbone == 'ViT-L/14':
                output_features = 768
            elif opt.vit_backbone in ['RN50']:
                output_features = 1024

        predicted_embd = np.zeros([n_samples, output_features], 'float32')
        true_embd = np.zeros([n_samples, output_features], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1
        
        data_iterator = tqdm(test_dataloader)
        it = 0
        for data in data_iterator:
            video, label, embd, idx, seen = data
            if len(video) == 0:
                continue

            # Run network on batch
            pred_embd = model(video.to(opt.device))

            pred_embd_np = pred_embd.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            predicted_embd[it:it + len(label)] = pred_embd_np
            true_embd[it:it + len(label)] = embd.squeeze()
            true_label[it:it + len(label)] = label.squeeze()
            good_samples[it:it + len(label)] = True
            it += len(label)

            #for validation loss
            embd = embd.to(opt.device)
            val_loss = criterion(pred_embd, embd)
            val_losses.append(val_loss.item())

    predicted_embd = predicted_embd[:it]
    true_embd, true_label = true_embd[:it], true_label[:it]

    #val_loss added
    val_loss = np.mean(val_losses)

    class_embedding = test_dataloader.dataset.class_embd
    accuracy, accuracy_top5, classwise_accuracy = compute_accuracy(predicted_embd, class_embedding, true_embd)

    with open(opt.results_path, 'a') as f:
        f.write('Test Accuracy = %.2f, Accuracy top-5 = %.2f, Val loss = %.4f\n' % (accuracy, accuracy_top5, val_loss))
        for i, acc in enumerate(classwise_accuracy):
            f.write('Accuracy of class %d is %.2f\n' % (i, acc))
    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(accuracy, f)
        pickle.dump(accuracy_top5, f)
        pickle.dump(classwise_accuracy, f)

    #returned val loss
    return accuracy, accuracy_top5, val_loss


def gzsl_test(test_dataloader, model, opt):
    # Perform GZSL testing.
    model.eval()
    with torch.no_grad():
        n_samples = len(test_dataloader.dataset)
        
        if semantic == 'word2vec' or semantic == 'fasttext':
            output_features = 300
        elif semantic == 'sent2vec':
            output_features = 600
        elif 'clip' in semantic:
            output_features = 512

        predicted_embd = np.zeros([n_samples, output_features], 'float32')
        true_embd = np.zeros([n_samples, output_features], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1
        seens = np.zeros((n_samples), 'bool')
        
        data_iterator = tqdm(test_dataloader)
        it = 0
        for data in data_iterator:
            video, label, embd, idx, seen = data
            if len(video) == 0:
                continue
            video = video.to(opt.device)

            # Run network on batch
            pred_embd = model(video.to(opt.device))
            pred_embd_np = pred_embd.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            predicted_embd[it:it + len(label)] = pred_embd_np
            true_embd[it:it + len(label)] = embd.squeeze()
            true_label[it:it + len(label)] = label.squeeze()
            good_samples[it:it + len(label)] = True
            seens[it:it + len(label)] = seen
            it += len(label)

    predicted_embd = predicted_embd[:it]
    true_embd, true_label = true_embd[:it], true_label[:it]

    class_embedding = test_dataloader.dataset.class_embd
    accuracy, accuracy_seen, accuracy_unseen, classwise_accuracy = compute_generalized_accuracy(predicted_embd, class_embedding, true_embd, seens)

    with open(opt.results_path, 'a') as f:
        f.write('Test Accuracy = %.2f, Seen accuracy = %.2f, Unseen accuracy = %.2f\n' % (accuracy, accuracy_seen, accuracy_unseen))
        for i, acc in enumerate(classwise_accuracy):
            f.write('Accuracy of class %d is %.2f\n' % (i, acc))
    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(accuracy, f)
        pickle.dump(classwise_accuracy, f)
    return accuracy
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':
    
    st = datetime.datetime.now()
    run_name = opt.novelty + '_' + opt.dataset + str(opt.split_index) + '_' + opt.action
    run = wandb.init(project="ZSAR_LCA", config=opt, name=run_name)

    #get config params
    trained_weights = wandb.config['trained_weights']
    seed = wandb.config['random_seed']
    network = wandb.config['network']
    save_path = wandb.config['save_path']
    dataset_name = wandb.config['dataset']
    semantic = wandb.config['semantic']
    lr = wandb.config['lr']
    n_epochs = wandb.config['n_epochs']
    batch_size = wandb.config['batch_size']

    LCA_drops = [0.0, 0.0, 0.0]


    opt.LCA_drops = LCA_drops


    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if trained_weights:
        if network == 'c3d':
            opt.model_path = '../datasets/pretrain_models/c3d/zsl_r2plus1d18_kinetics700_ucf101_hmdb51_checkpoint.pth.tar'
        elif network == 'r2plus1d':
            opt.model_path = '../datasets/pretrain_models/r2plus1d_18/zsl_r2plus1d18_kinetics700_ucf101_hmdb51_checkpoint.pth.tar'
    else:
        opt.last_model_path = save_path + '/checkpoint_last_epoch.pth.tar'
        opt.model_path = save_path + '/checkpoint.pth.tar'
    opt.model_kg_path = save_path + '/checkpoint_kg.pth.tar'

    opt.results_path = save_path + opt.action + '_' + dataset_name + '_accuracy.txt'
    opt.pickle_path = save_path + opt.action + '_' + dataset_name + '.pickle'

    # Create logging directory
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True

    # Set options
    if torch.cuda.is_available():
        opt.device = 'cuda'
    elif torch.has_mps:
        opt.device = 'mps'
    else:
        opt.device = 'cpu'

    with open(opt.results_path, 'a') as f:
        f.write("New run started\n")
        if torch.cuda.device_count() > 1:
            f.write("Let's use", torch.cuda.device_count(), "GPUs!\n")
        f.write(str(opt) + "\n")

    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(opt, f)

    # Load datasets
    dataloaders, all_seen_classes, all_unseen_classes = dataset.load_datasets(opt)

    model = visual_network.get_network(opt)

    train_actions = ['train', 'fsl_train', 'sup_train']
    test_actions = ['test', 'gzsl_test', 'fsl_test', 'sup_test']

    if opt.action in train_actions:
        if os.path.isfile(opt.last_model_path):
            opt.weights = opt.last_model_path 
        elif os.path.isfile(opt.model_path):
            opt.weights = opt.model_path

    else:
        #For testing, always load the best model
        if os.path.isfile(opt.model_path):
            opt.weights = opt.model_path

    epoch_done = -1
    if opt.weights and opt.weights != "none":
        j = len('module.')
        weights = torch.load(opt.weights)['state_dict']
        model_dict = model.state_dict()
        weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        with open(opt.results_path, 'a') as f:
            f.write("LOADED MODEL:  " + opt.weights + "\n")
        # Resuming epoch number
        if 'epoch' in torch.load(opt.weights):
            epoch_done = torch.load(opt.weights)['epoch']

    if opt.count_params:
        count_parameters(model)
        exit()
        
    model = torch.nn.DataParallel(model)
    model.to(opt.device)
    criterion = torch.nn.MSELoss().to(opt.device)

    if opt.action == 'train' or opt.action == 'sup_train':
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)

        best_accuracy = 0
        best_accuracy_epoch = -1
        for epoch in range(epoch_done + 1, n_epochs):
            model.train()
            train_accuracy, train_loss = train(dataloaders['training'][0], model, optimizer, criterion, opt, epoch)
            wandb.log({
                'epoch': epoch+1,
                'train_acc': train_accuracy,
                'train_loss': train_loss
            })
            
            # Save latest model
            torch.save({
                'epoch':epoch,
                'state_dict': model.state_dict(), 
                'opt': opt, 
                'train_accuracy': train_accuracy,
                'train_loss': train_loss},
                opt.last_model_path)
            
            if opt.no_val == False:
                if epoch % opt.val_freq == (opt.val_freq - 1):
                    val_accuracy, _, val_loss = test(dataloaders['validation'][0], model, opt, criterion)
                    
                    wandb.log({
                        'epoch': epoch+1,
                        'val_acc': val_accuracy,
                        'val_loss': val_loss
                    })

                    if val_accuracy > best_accuracy:
                        # Save best model
                        torch.save({
                            'epoch':epoch,
                            'state_dict': model.state_dict(), 
                            'opt': opt, 
                            'accuracy': val_accuracy,
                            'val_loss': val_loss},
                            opt.model_path)
                        best_accuracy = val_accuracy
                        best_accuracy_epoch = epoch

                    # Early stopping
                    elif epoch - best_accuracy_epoch > opt.early_stop_thresh:
                        print('\nEarly stopping.....')
                        break
            else:
                if train_accuracy > best_accuracy:
                    # Save best model
                    torch.save({'epoch':epoch, 'state_dict': model.state_dict(), 'opt': opt, 'accuracy': train_accuracy, 'train_loss': train_loss},
                            opt.model_path)
                    best_accuracy = train_accuracy

            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

    if opt.action == 'test' or opt.action == 'sup_test':
        for test_dataloader in dataloaders['testing']:
            test(test_dataloader, model, opt, criterion)

    if opt.action == 'gzsl_test':
        for test_dataloader in dataloaders['testing']:
            gzsl_test(test_dataloader, model, opt)

    et = datetime.datetime.now()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'H:M:S')
