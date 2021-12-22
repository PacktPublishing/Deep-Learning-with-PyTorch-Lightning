import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pk_pdd_seq
import torchvision.models as models

import pytorch_lightning as pl

class HybridModel(pl.LightningModule):
    def __init__(self, cnn_embdng_sz, lstm_embdng_sz, lstm_hidden_lyr_sz, lstm_vocab_sz, lstm_num_lyrs, max_seq_len=20):
        super(HybridModel, self).__init__()

        resnet = models.resnet152(pretrained=False)
        module_list = list(resnet.children())[:-1]
        self.cnn_resnet = nn.Sequential(*module_list)
        self.cnn_linear = nn.Linear(resnet.fc.in_features,
                                    cnn_embdng_sz)
        self.cnn_batch_norm = nn.BatchNorm1d(cnn_embdng_sz,
                                             momentum=0.01)

        self.lstm_embdng_lyr = nn.Embedding(lstm_vocab_sz,
                                            lstm_embdng_sz)
        self.lstm_lyr = nn.LSTM(lstm_embdng_sz,
                                lstm_hidden_lyr_sz,
                                lstm_num_lyrs,
                                batch_first=True)
        self.lstm_linear = nn.Linear(lstm_hidden_lyr_sz,
                                     lstm_vocab_sz)
        self.max_seq_len = max_seq_len
        self.save_hyperparameters()

    def forward(self, input_images, caps, lens):
        cnn_features = self.cnn_batch_norm(self.forward_cnn_no_batch_norm(input_images))

        embeddings = self.lstm_embdng_lyr(caps)
        embeddings = torch.cat((cnn_features.unsqueeze(1), embeddings), 1)
        lstm_input = pk_pdd_seq(embeddings, lens, batch_first=True)
        hddn_vars, _ = self.lstm_lyr(lstm_input)
        model_outputs = self.lstm_linear(hddn_vars[0])
        return model_outputs

    def forward_cnn_no_batch_norm(self, input_images):
        with torch.no_grad():
            features = self.cnn_resnet(input_images)
        features = features.reshape(features.size(0), -1)
        return self.cnn_linear(features)

    def configure_optimizers(self):
        params = list(self.lstm_embdng_lyr.parameters()) + \
                 list(self.lstm_lyr.parameters()) + \
                 list(self.lstm_linear.parameters()) + \
                 list(self.cnn_linear.parameters()) + \
                 list(self.cnn_batch_norm.parameters())
        optimizer = torch.optim.Adam(params, lr=0.0003)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss_criterion = nn.CrossEntropyLoss()
        imgs, caps, lens = batch
        outputs = self(imgs, caps, lens)
        targets = pk_pdd_seq(caps, lens, batch_first=True)[0]
        loss = loss_criterion(outputs, targets)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def get_caption(self, img, lstm_sts=None):
        features = self.forward_cnn_no_batch_norm(img)

        token_ints = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hddn_vars, lstm_sts = self.lstm_lyr(inputs, lstm_sts)
            model_outputs = self.lstm_linear(hddn_vars.squeeze(1))
            _, predicted_outputs = model_outputs.max(1)
            token_ints.append(predicted_outputs)
            inputs = self.lstm_embdng_lyr(predicted_outputs)
            inputs = inputs.unsqueeze(1)
        token_ints = torch.stack(token_ints, 1)
        return token_ints
