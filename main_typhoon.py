import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import init
from typhoon_dataset import TyphoonDataset
from model import TCN, Rnn, Gru
from torch.optim import lr_scheduler
import argparse
import json



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains TCN on Typhoon', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
    parser.add_argument('--decay', '-d', type=float, default=1e-3, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=64)
    parser.add_argument('--save', '-s', type=str, default='./best/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    parser.add_argument('--log', type=str, default='./log', help='Log folder.')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # train valid split
    dataset = TyphoonDataset(mode="train")
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                                        num_workers=args.prefetch, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_bs, shuffle=False,
                                                                        num_workers=args.prefetch, pin_memory=False)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = TCN(22, 2, 1, [64, 128, 256, 1024], 2, 0.2, False)
    # net = Rnn(22, 64, 4, 2, drop=0.2)
    # net = Gru(22, 64, 4, 2, drop=0.2)

    if(args.load!=None):
        net.load_state_dict(torch.load(args.load))

    net.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3, factor=0.9)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    # criterion = nn.MSELoss().cuda()

#######################TRAIN TEST FUNCTION########################################################################

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for stpe, (_, trace, y) in enumerate(train_loader):
            trace, y = trace.cuda(), y.cuda()

            # forward
            output = net(trace)

            # backward
            optimizer.zero_grad()
            loss = F.mse_loss(output, y)
            del output
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg += loss.item()

        state['train_loss'] = loss_avg / len(train_loader)


    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        distance = 0.0
        for step, (_, trace, y) in enumerate(valid_loader):
            trace, y = trace.cuda(), y.cuda()

            # forward
            output = net(trace)
            loss = F.mse_loss(output, y)

            # distance
            length = y.shape[0]
            dealt = torch.pow((output-y), 2)
            del output
            distance += torch.sum(torch.sqrt(torch.sum(dealt.cpu(), dim=1))).item() / length
            # test loss average
            loss_avg += loss.item()

        state['test_loss'] = loss_avg / len(valid_loader)
        state['distance'] = distance / len(valid_loader)

#####################TRAIN TEST FUNCTION################################################################################

    # Main loop
    best_distance = 100
    for epoch in range(args.epochs):
        # current_lr = lr0 / 2**int(epoch/50)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr
        state['learning_rate'] = optimizer.param_groups[0]['lr']
        state["epoch"] = epoch
        train()
        test()
        scheduler.step(state['test_loss'])
        # scheduler.step()
        if state["distance"] < best_distance:
            best_distance = state["distance"]
            torch.save(net.state_dict(), os.path.join(args.save, 'model_{}.pytorch'.format(best_distance)))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best distance: {}".format(best_distance))
    
    log.close()
