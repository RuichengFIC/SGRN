from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import *
import numpy as np
import torch
import random
import copy
import gc

def train(model, x_train, y_train, lr =0.001, MAX_EPOCH = 50, check_mode = False, log_file = False):
    model.train()
    sample_num = len(x_train)
    train_seq = list(range(sample_num))
    batch_size = 100
    
#     max_patience = 15
#     patience = 0

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    min_loss = 1000000
    
    for epoch in range(MAX_EPOCH):
        random.shuffle(train_seq)
        total_loss = 0
        for batch, i in enumerate(range (0, sample_num, batch_size)):
            optimizer.zero_grad()
            pred = model(x_train[train_seq[i: i+batch_size]])
            loss = loss_fn(pred, y_train[train_seq[i: i+batch_size]])
            # Backpropagation
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
               
        if total_loss < min_loss:
            patience = 0
            best_epoch = epoch
            min_loss = total_loss
            best_state_dict = copy.deepcopy(model.state_dict())
#             model.load_state_dict(state_dict)
         
#         patience = patience + 1       
#         if patience > max_patience:
#             print("break")
#             break 
            
        if check_mode:
            if epoch % 10 == 0:
                print(epoch, total_loss)
        if log_file and epoch % 10 == 0:
            with open(log_file,"a") as fo:
                fo.write("epoch: {}, total_loss: {:.3f} \n".format(epoch, total_loss))
        gc.collect()
        torch.cuda.empty_cache()
        
    model.load_state_dict(best_state_dict)
    if check_mode:
        print(best_epoch, min_loss)

def evaluate(model, x_test, y_test, check_mode = False, log_file = False):
    model.eval()
    pred = model(x_test)
    pred_class = pred.argmax(-1).detach().cpu().numpy()
    target_names = ["{}".format(n) for n in np.unique(y_test)]
    
    res = classification_report(y_test, pred_class, target_names=target_names,output_dict=True, zero_division = 1)
    acc, precision, recall, f1_socre = (res["accuracy"], res["weighted avg"]["precision"], res["weighted avg"]["recall"], res["weighted avg"]["f1-score"])
    if log_file:
        with open(log_file,"a") as fo:
            fo.write("acc:{:.3f}, f1_score: {:.3f} \n".format(acc,f1_socre))
    if check_mode:
        print(classification_report(y_test, pred_class, target_names=target_names, zero_division = 1))
#     auc = metrics(y_test.detach().cpu().numpy(),pred.detach().cpu().numpy())
#     print(auc)
    return acc, precision, recall, f1_socre


def batch_eval(eval_model, args, x, y, lr=0.0005, times = 5, MAX_EPOCH=50, check_mode=False, log_file = False):
    DEVICE = args["DEVICE"]
    accs = []
    precisions = []
    recalls = []
    f1s = []
    
    kf = KFold(n_splits= times,shuffle= True, random_state=1)

    kf.get_n_splits(x)
    
    for train_index, test_index in kf.split(x):
        model = eval_model(ipt_dim=args["input_dim"], hid_dim=args["hid_dim"], num_class=args["num_class"], use_MRS = args["use_MRS"])
        model.to(torch.double)
        model.to(DEVICE)
  
        train_x = x[train_index]
        test_x = x[test_index]
        train_x = torch.tensor(train_x, dtype=torch.double,device = DEVICE)
        test_x = torch.tensor(test_x, dtype=torch.double,device = DEVICE)
            
        train_y = y[train_index]
        test_y = y[test_index]
        train_y = torch.tensor(train_y, dtype=torch.long, device = DEVICE)

        
        train(model, train_x, train_y, MAX_EPOCH= MAX_EPOCH, lr = lr, check_mode = check_mode, log_file = log_file)
        acc, precision, recall, f1_socre = evaluate(model,test_x,test_y, check_mode = check_mode, log_file = log_file)
        
        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1_socre)
    if log_file:
        with open(log_file,"a") as fo:
            fo.write("acc: {:.3f}({:.3f}), f1_score: {:.3f}({:.3f}) \n".format(np.mean(accs),np.std(accs),np.mean(f1s),np.std(f1s)))
    print(np.mean(accs),np.std(accs),np.mean(f1s),np.std(f1s))