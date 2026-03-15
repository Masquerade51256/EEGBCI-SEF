import config
from Dataloader import dataloader
from Models.get_model import get_model
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import logging

train_device = torch.device(config.train_device if torch.cuda.is_available() else "cpu")
print(train_device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train(model, data):
    wd = config.weight_decay
    lr = config.learning_rate
    epoch_num = config.num_epochs
    batch_size = config.batch_size
    kf = KFold(n_splits=config.k_folds,shuffle=True,random_state=0)
    print(f"--> train begin")

    k = 0
    best_val_acc = 0.0
    best_model_state = None

    for train_index, val_index in kf.split(data):
        print(f"-->fold: {k}")
        k += 1

        # split
        data_train = torch.utils.data.dataset.Subset(data,train_index)
        data_val = torch.utils.data.dataset.Subset(data,val_index)
        dataloader_train = DataLoader(data_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=train_device))
        dataloader_val = DataLoader(data_val, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=train_device))

        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)

        fold_best_val_acc = 0.0

        for epoch in range(epoch_num):
            # cosine annealing
            lr = (1 + math.cos(epoch * math.pi / epoch_num)) * config.learning_rate / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_loss = 0
            train_correct = 0
            train_size = 0
            for images,labels in dataloader_train:
                images = images.to(train_device)
                print(f"images: {images.shape} ")
                labels = labels.to(train_device)

                optimizer.zero_grad()
                outputs = model(images)
                print(outputs.shape)
                print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_size += labels.size(0)
            
            avg_train_loss = train_loss / len(dataloader_train)
            train_acc = train_correct / train_size

            # print(f"--> validation begin")

            model.eval()
            val_loss = 0
            val_correct = 0
            val_size = 0

            with torch.no_grad():
                for val_images, val_labels in dataloader_val:
                    val_images = val_images.to(train_device)
                    val_labels = val_labels.to(train_device)
                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_correct += (val_predicted == val_labels).sum().item()
                    val_size += val_labels.size(0)
            
            avg_val_loss = val_loss / len(dataloader_val)
            val_acc = val_correct / val_size
            print(f"epoch: {epoch:02d}   train loss: {train_loss:.4f}   train acc: {(train_correct/train_size):.2f}   val loss: {val_loss:.4f}   val acc: {(val_correct/val_size):.2f}")

            logger.info(f"Fold {k-1}, Epoch {epoch:03d}: LR={lr:.6f}, "
                       f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                       f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if val_acc > fold_best_val_acc:
                fold_best_val_acc = val_acc
                
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'train_acc': train_acc,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                    'fold': k-1
                }
                # 构建包含关键信息的文件名
                ckpt_filename = f"best_model_{config.SELECTED_DATASET}_{config.SELECTED_MODEL}_fold_{k-1}_acc_{fold_best_val_acc:.4f}.pt"
                torch.save(checkpoint, ckpt_filename)
                logger.info(f"  -> Fold {k-1} 最佳模型已保存: {ckpt_filename} (准确率: {fold_best_val_acc:.4f})")
                
                # 同时更新全局最佳模型（这里简单取最后一个fold的最佳模型，也可改为跟踪所有fold中最好的）
                best_val_acc = fold_best_val_acc
                best_model_state = model_to_save.state_dict()
        
        logger.info(f"--- Fold {k-1} finished, highest acc: {fold_best_val_acc:.4f} ---")
    
    # 加载最终的最佳模型状态（此处返回最后一个fold的最佳模型，您可以根据需求修改策略）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"已加载最终模型权重（验证准确率: {best_val_acc:.4f}）。")
    return model
            

def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data:
            images = images.to(train_device)
            labels = labels.to(train_device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # load dataset
    dataset = dataloader.load_data(config.SELECTED_DATASET)
    # build model
    model = get_model(config.SELECTED_MODEL)
    # train model
    trained_model = train(model, dataset)
    # evaluate model
    # result = evaluate(trained_model, dataset)
    # visualize results
    # print(result)
    pass


if __name__ == "__main__":
    main()