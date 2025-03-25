import torch
import torch.nn as nn
import torch.optim as optim
from transformers import MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class NMTModel(nn.Module):
    def __init__(self, model_name):
        super(NMTModel, self).__init__()
        from transformers import MarianMTModel
        self.model = MarianMTModel.from_pretrained(model_name)
        
        # total_params = list(self.model.parameters())
        # num_freeze = int(0.7 * len(total_params))
        # for param in total_params[:num_freeze]:
        #     param.requires_grad = False

        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

        num_layers_to_freeze = 4
        for i in range(num_layers_to_freeze):
            for param in self.model.model.decoder.layers[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def predict(self, input_ids, attention_mask, max_length=100):
        with torch.no_grad():
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
    

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=16, lr=1e-3, epochs=20, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.save_path = '/content/drive/MyDrive/ft_marian.pt'


    def train(self):
        self.model.train()
        train_losses = []
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch", position=0, leave=True)

            for batch in progress_bar:
                en_input_ids = batch["en_input_ids"].to(self.device)
                en_attention_mask = batch["en_attention_mask"].to(self.device)
                hi_input_ids = batch["hi_input_ids"].to(self.device)
                hi_attention_mask = batch["hi_attention_mask"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(en_input_ids, attention_mask=en_attention_mask, labels=hi_input_ids)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(self.train_dataloader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss}")

            if self.val_dataloader:
                self.validate()
        
        print("Training Complete.")
        torch.save(self.model.state_dict(), self.save_path)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plot_path = self.save_path.replace('.pt', '_loss_plot.png')  
        plt.savefig(plot_path)
        plt.close()
    

    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                en_input_ids = batch["en_input_ids"].to(self.device)
                en_attention_mask = batch["en_attention_mask"].to(self.device)
                hi_input_ids = batch["hi_input_ids"].to(self.device)
                hi_attention_mask = batch["hi_attention_mask"].to(self.device)
                
                outputs = self.model(en_input_ids, attention_mask=en_attention_mask, labels=hi_input_ids)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_val_loss = total_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss}")
        self.model.train()
