import torch
class ModelManagement:
    def __init__(self, path, name_model):
        self.path = path
        self.last_metrics = 10**8
        self.name_model = name_model
        self.dict_model = None

    def checkpoint(self, epoch, model, optimizer, loss):
        if self.last_metrics > loss:
            self.dict_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            self.last_metrics = loss
            self.save_best_model()

    def save_best_model(self):
        torch.save(self.dict_model, self.path + '%s.pth' % self.name_model)
