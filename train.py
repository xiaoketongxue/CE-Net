import socket
import timeit
from datetime import datetime
import os
import glob
import numpy as np
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders.utils import extract_each_layer

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, sbd, combine_dbs
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from dataloaders import custom_transforms as tr
from data import ImageFolder

import loss
import scipy.misc as misc
from Visualizer import Visualizer


viz = Visualizer(env="Deeplab-V3")

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False  # Whether to use SBD dataset
nEpochs = 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 10  # Training batch size
testBatch = 1  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-3  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.99  # Momentum
p['epoch_size'] = 10  # How many epochs to change learning rate
backbone = 'resnet' # Use xception or resnet as feature extractor,

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run')

# Network definition
if backbone == 'xception':
    net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=12, os=16, pretrained=False)
elif backbone == 'resnet':
    net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=12, os=16, pretrained=False)
else:
    raise NotImplementedError

modelName = 'deeplabv3plus-' + backbone + '-voc'
criterion = loss.dice_bce_loss()
# criterion = loss.FocalLoss(gamma=2)

print("Backbone network is {}".format(backbone))

if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))


if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    # if pretrain model
    # ignored_params = list(map(id, net.module.xception_features.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      net.parameters())
    #
    # optimizer = torch.optim.Adam([
    #     {'params': base_params},
    #     {'params': net.module.xception_features.parameters(), 'lr': p['lr'] * 0.1}
    # ], lr=p['lr'], weight_decay=p['wd'])

    optimizer = optim.Adam(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    # voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    # voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

    voc_train = ImageFolder(root_path='dataset/train', datasets='OCT-origin')
    voc_val = ImageFolder(root_path='dataset/validation', datasets='OCT-origin', mode='test')

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = voc_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)

    # utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")
    min_mean_error = 1000.0

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        # print(epoch)
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)

            # conv5_params = list(map(id, net.conv5.parameters()))
            # conv4_params = list(map(id, net.conv4.parameters()))
            # base_params = filter(lambda p: id(p) not in conv5_params + conv4_params,
            #                      net.parameters())

            # optimizer = torch.optim.SGD([
            #     {'params': base_params},
            #     {'params': net.conv5.parameters(), 'lr': lr * 100},
            #     {'params': net.conv4.parameters(), 'lr': lr * 100}]
            #     , lr=lr, momentum=0.9)

            optimizer = optim.Adam(net.parameters(), lr=p['lr'], weight_decay=p['wd'])

        net.train()


        for ii, sample_batched in enumerate(trainloader):

            # inputs, labels = sample_batched['image'], sample_batched['label']
            inputs, labels = sample_batched[0], sample_batched[1]
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]


            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)


            # loss = criterion(labels, outputs)
            loss = criterion(labels, outputs)
            viz.plot(name='loss', y=loss.data[0])

            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                # writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            # if ii % (num_img_tr // 10) == 0:
            #     grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
            #     # writer.add_image('Image', grid_image, global_step)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
            #                            range=(0, 255))
            #     # writer.add_image('Predicted label', grid_image, global_step)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
            #     # writer.add_image('Groundtruth label', grid_image, global_step)

        # One testing epoch
        # if useTest and epoch % nTestInterval == (nTestInterval - 1):
        if useTest:
        # if useTest and epoch > 5 and epoch % 5 == 0:
            k = 0
            all_mse = []
            # total_miou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched[0], sample_batched[1]

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.nn.Softmax(dim=1)(outputs)

                # print(predictions)

                # loss = criterion(outputs, labels, size_average=False, batch_average=True)
                # running_loss_ts += loss.item()
                labels = labels.cpu().data.numpy()
                labels = np.squeeze(labels, axis=[0, 1])


                predictions = predictions.cpu().data.numpy()
                predictions = np.squeeze(np.argmax(predictions, 1), axis=0)

                images = inputs.cpu().data.numpy()
                images = np.squeeze(images, axis=0)[0, :, :]
                # print(np.shape(images))
                # images = np.squeeze(images, axis=0)

                show_image = np.zeros(shape=(512, 1710))
                show_image[:, :512] = labels
                show_image[:, 1100-512:1100] = predictions
                show_image[:, 1710-512:1710] = images

                misc.imsave(os.path.join('show_domo', str(ii) + '.png'), show_image)


                # print(np.shape(labels), np.min(labels), np.max(labels))
                # print(np.shape(predictions), np.min(predictions), np.max(predictions))

                mse = np.zeros(shape=[1, 10])
                k = k + 1

                for i in range(10):
                    layer_one = extract_each_layer(predictions.astype(np.uint8), i + 0.5)
                    layer_one_gt = extract_each_layer(labels.astype(np.uint8), i + 0.5)
                    mse[0, i] = np.mean(np.abs(np.subtract(layer_one, layer_one_gt)))


                each_error = np.mean(mse, axis=0)
                # print(each_error)
                all_mse.append(each_error)
                # print(all_mse)
            mean_error = np.mean(all_mse, axis=0)
            viz.log(info=mean_error, win='mean_error')
            print('mean error is {}'.format(mean_error))

            mean_error_mean = np.mean(mean_error)
                # Print stuff
                # if ii % num_img_ts == num_img_ts - 1:
                #
                #     miou = total_miou / (ii * testBatch + inputs.data.shape[0])
                #     running_loss_ts = running_loss_ts / num_img_ts
                #
                #     print('Validation:')
                #     print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                #     # writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                #     # writer.add_scalar('data/test_miour', miou, epoch)
                #     print('Loss: %f' % running_loss_ts)
                #     print('MIoU: %f\n' % miou)
                #     running_loss_ts = 0


            if mean_error_mean < 1.8:
            # Save the model
            #     min_mean_error = mean_error_mean
                viz.log(info=mean_error_mean, win='mean_error_mean')

                torch.save(net.state_dict(), os.path.join(save_dir, 'best_models-' + str(epoch) + '.pth'))
                print("Save model at {}\n".format(os.path.join(save_dir, 'best_models-' + str(epoch) + '.pth')))

