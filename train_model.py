'''The following module trains the weights of the neural network model.'''
import os
import datetime
import uuid
from tqdm.auto import tqdm
import glob

import torch
import torch.nn    as nn
import torch.optim as optim

from data_declaration import Task
from loader_helper    import LoaderHelper
from architecture     import load_cam_model, Camull
from evaluation import evaluate_model

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

def save_weights(model_in, uuid_arg, fold=1, task: Task = None):
    '''The following function saves the weights file into required folder'''
    root_path = ""

    if task == Task.NC_v_AD:
        root_path = "/content/weights/NC_v_AD/"     + uuid_arg + "/"
    else:
        root_path = "/content/weights/sMCI_v_pMCI/" + uuid_arg + "/"

    if fold == 1:
        os.makedirs(root_path,exist_ok=True)
     ###########   os.mkdir(root_path) #otherwise it already exists

    while True:

        s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format(fold, date=datetime.datetime.now()) # pylint: disable=line-too-long

        if os.path.exists(s_path):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break

def load_model(arch, path=None):
    '''Function for loaded camull net from a specified weights path'''
    if arch == "camull": #must be camull

        if path is None:
            model = load_cam_model("/content/weights/CN_v_AD/cec0b7c0dfbf4f439600f5edeb45320e/fold_1_weights")
        else:
            model = load_cam_model(path)
            
    return model

def build_arch():
    '''Function for instantiating the pytorch neural network object'''
    net = Camull()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(DEVICE)
    net.double()

    return net


def train_loop(model_in, train_dl, epochs):
    '''Function containing the neural net model training loop'''
    optimizer = optim.Adam(model_in.parameters(), lr=0.001, weight_decay=5e-5)

    loss_function = nn.BCELoss()

    model_in.train()

    for i in range(epochs):

        for _, sample_batched in enumerate(tqdm(train_dl)):

            batch_x = sample_batched['mri'].to(DEVICE)
            batch_xb = sample_batched['clin_t'].to(DEVICE)
            batch_y = sample_batched['label'].to(DEVICE)

            model_in.zero_grad()
            outputs = model_in((batch_x, batch_xb))

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss.item(), 5)))


def train_camull(ld_helper, k_folds=5, model=None, epochs=40):
    '''The function for training the camull network'''
    task = ld_helper.get_task()
    uuid_ = uuid.uuid4().hex
    model_cop = model
  #  print("##################################################print(model_cop)")
  #  print(model_cop)
    for k_ind in range(k_folds):

        if model_cop is None:
            model = build_arch()
        else:
            model = model_cop
    #    print("##################################################print(model)")
    #    print(model)
        train_dl = ld_helper.get_train_dl(k_ind)
        print("train_dl=",train_dl)
        train_loop(model, train_dl, epochs)
        save_weights(model, uuid_, fold=k_ind+1, task=task)

        print("Completed fold {}/{}.".format(k_ind, k_folds))
    train_camull.variable = uuid_
    return uuid_

def start(ld_helper, epochs=40, model_uuid=None):

    task = ld_helper.get_task()
    if (task == Task.NC_v_AD):
        model_uuid = train_camull(ld_helper, epochs=40)
    else: # sMCI vs pMCI
        if (model_uuid != None):
            model = load_model("camull", model_uuid)
            model_uuid  = train_camull(ld_helper, model=model, epochs=40)
        else:
            print("Need a model uuid")
    
    return model_uuid


def main():
    ld_helper = LoaderHelper(task=Task.NC_v_AD)
    model_uuid = train_camull(ld_helper, epochs=40)
    evaluate_model(DEVICE, train_camull.variable, ld_helper)
   # ld_helper.change_task(Task.sMCI_v_pMCI)
    ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)
    past_one="/content/weights/NC_v_AD/" + train_camull.variable
    past_two=past_one+"/"+"*"
    past_three=glob.glob(past_two)
    model = load_model("camull", past_three[0])
    model_uuid = train_camull(ld_helper, epochs=40)
    evaluate_model(DEVICE, train_camull.variable, ld_helper)
    
    '''Main function of the module.'''
    #NC v AD
  #@  ld_helper = LoaderHelper(task=Task.NC_v_AD)
  #@  model_uuid = train_camull(ld_helper, epochs=1)
    #print(train_camull.variable)
    #print(type(train_camull.variable))
  #@  evaluate_model(DEVICE, train_camull.variable, ld_helper)

    #transfer learning for pMCI v sMCI
  #@  ld_helper.change_task(Task.sMCI_v_pMCI)
  #@  past_one="/content/weights/NC_v_AD/" + train_camull.variable
 #   print(past_one)
  #@  past_two=past_one+"/"+"*"
 #   print(past_two)
  #@  past_three=glob.glob(past_two)
 #   print(past_three[0])
  #@  model = load_model("camull", past_three[0])
 #   print(model)
  #@  uuid  = train_camull(ld_helper, model=model, epochs=1)
  #@  evaluate_model(DEVICE, train_camull.variable, ld_helper)
     

main()
