import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
import wandb
# from torchmetrics.classification import MulticlassAccuracy

# Device Agnostic code
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device):
  
  train_loss, train_acc = 0, 0

  model.train()


  for batch, (x_train, y_train) in enumerate(dataloader):

    # if device == 'cuda':
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    # print()
    # print(x_train.dtype)
    # print("----------------------------------------------------------------------------------------")
    # print(y_train, y_train.dtype)
    # print("----------------------------------------------------------------------------------------")
    # # print(model.dtype)
    # print("----------------------------------------------------------------------------------------")
    # 1. Forward step
    pred = model(x_train)
    
    # print("\n", pred, pred.dtype)
    # print("\n", torch.argmax(pred, dim=1), torch.argmax(pred, dim=1).dtype)
    # print("\n", pred.shape, y_train.shape, torch.argmax(pred, dim=1).shape)

    # 2. Loss
    # print(pred.shape)
    # print(y_train.shape)
    loss = loss_fn(pred, y_train)

    # 3. Grad zerostep
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    acc = accuracy_fn(torch.argmax(pred, dim=1), y_train)

    train_loss += loss
    train_acc += acc



  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  # print(train_loss, train_acc)
  return train_loss, train_acc, model


# test
def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, device: torch.device):
  
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:
      
      # if device == 'cuda':
      x_test, y_test = x_test.to(device), y_test.to(device)

      # 1. Forward
      test_pred = model(x_test)
      
      # 2. Loss and accuray
      # print(test_pred)
      # print(y_test)
      test_loss += loss_fn(test_pred, y_test)
      
      acc = accuracy_fn(torch.argmax(test_pred, dim=1), y_test)
      test_acc += acc

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  # print(train_loss, train_acc)
  return test_loss, test_acc



def eval_func(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, device: torch.device):
  
  eval_loss, eval_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for x_eval, y_eval in dataloader:
      
      if device == 'cuda':
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

      # 1. Forward
      eval_pred = model(x_eval)
      
      # 2. Loss and accuray
      eval_loss += loss_fn(eval_pred, y_eval)

      acc = accuracy_fn(torch.argmax(eval_pred, dim=1), y_eval)
      eval_acc += acc


    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)

  # print(eval_loss, eval_acc)
  return eval_loss, eval_acc


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device: torch.device, args):
  """
  Train and test the pytorch model.
  
  Args:
    model: A pytorch model to be train and test
    train_dataloader: A dataloader instance for model training
    test_dataloader: A dataloader instance for model testing
    optimizer: A optimizer to optimize the model
    loss_fn: A loss function to calculate loss on both datasets
    epoch: An interger indication to how much train the model
    device: A device on which model to be train and test

  Return:
    List of train and test loss and accuracy in form of list with plot.
    train also return train model weights

  Example usage:
    train(model = model_0, train_dataloader = train_dataloader, test_dataloader = test_dataloader,
          optimizer = optim, loss_fn, loss_fn = loss_fn, epochs = n, device = device) 
  """

  # setup wandb
  wandb.init( project=args['wandb_project'], name=args['wandb_runname'], config=args)
  # wandb.require("core")
  
  train_losses, test_losses = [], []
  train_accs, test_accs = [], []
  for epoch in tqdm(range(epochs)):
    
    train_loss, train_acc, train_model = train_loop(model = model, dataloader = train_dataloader,
                                                    loss_fn = loss_fn, optimizer = optimizer,
                                                    accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = test_loop(model = model, dataloader = test_dataloader, loss_fn = loss_fn,
                                    accuracy_fn = accuracy_fn, device = device)
    
    wandb.log({
      "Training Loss": train_loss,
      "Test Loss": test_loss,
      "Training Accuracy": train_acc,
      "Test Accuracy": test_acc
    })

    # if epoch % 10 == 0 and epoch != 0:
    #   print(f"Before Scheduler Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}\n")
    #   scheduler.step()
    #   print(f"After Scheduler Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}\n")
    
    print(f"\nEpoch: {epoch+1}")
    print(f"Train Loss: {train_loss:.5f}  Test Loss: {test_loss:.5f}  ||  Train Accuray: {train_acc:.5f}  Test Accuray: {test_acc:.5f}")
    
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())
    
  wandb.finish()

  return train_model, train_losses, test_losses, train_accs, test_accs