import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.utils import shuffle
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchsampler import ImbalancedDatasetSampler
from torchmetrics.functional import auroc

data_dir = '<PATH_TO_VINDR-MAMMO-DATASET>'

image_size = (512, 512)     # image input size (depends on data pre-processing)
val_percent = 0.1           # how much of total training samples are used for model selection (default 10%)
batch_size = 200            # batch size may need to be adjusted depending on GPU memory
epochs = 20                 # number of training epochs
num_workers = 4             # number threads for data processing


class MammoDataset(Dataset):
    def __init__(self, data, data_dir, image_size, augmentation=False):
        self.data = data.reset_index(drop=True)
        self.data_dir = data_dir        
        self.image_size = image_size
        self.do_augment = augmentation

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        ])

        # geometric data augmentation
        self.geometric_augment = T.Compose([
            T.RandomApply(transforms=[T.RandomAffine(degrees=10, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = os.path.join(self.data_dir, 'images_512x512', self.data.loc[idx, 'study_id'], self.data.loc[idx, 'image_id'] + '.png')
            img_label = np.array(self.data.loc[idx, 'malignancy_label'], dtype='int64')

            sample = {'image_path': img_path, 'label': img_label, 'study_id': self.data.loc[idx, 'study_id'], 'image_id': self.data.loc[idx, 'image_id']}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        image = image.repeat(3, 1, 1)            

        if self.do_augment:
            image = self.photometric_augment(image.type(torch.ByteTensor)).type(torch.FloatTensor)
            image = self.geometric_augment(image)        

        return {'image': image, 'label': label, 'study_id': sample['study_id'], 'image_id': sample['image_id']}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label'], 'study_id': sample['study_id'], 'image_id': sample['image_id']}
    
    def get_labels(self):
        labels = [int(sample['label']) for sample in self.samples]
        return labels


class MammoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_size, val_percent, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data = pd.read_csv(os.path.join(self.data_dir,'breast-level_annotations.csv'))
        self.data['malignancy_label'] = self.data['breast_birads']

        # Define positive and negatives based on BI-RADS categories
        self.data.loc[self.data['malignancy_label'] == 'BI-RADS 1', 'malignancy_label'] = 0
        self.data.loc[self.data['malignancy_label'] == 'BI-RADS 2', 'malignancy_label'] = 0
        self.data.loc[self.data['malignancy_label'] == 'BI-RADS 3', 'malignancy_label'] = 0
        self.data.loc[self.data['malignancy_label'] == 'BI-RADS 4', 'malignancy_label'] = 0
        self.data.loc[self.data['malignancy_label'] == 'BI-RADS 5', 'malignancy_label'] = 1

        # Use pre-defined splits to separate data into development and testing
        self.dev_data = self.data[self.data['split'] == 'training']
        self.test_data = self.data[self.data['split'] == 'test']

        # Split development data into training and validation (for model selection)
        # Making sure images from the same subject are within the same set
        unique_study_ids = self.dev_data.study_id.unique()

        unique_study_ids = shuffle(unique_study_ids)
        num_train = (round(len(unique_study_ids)*(1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids[num_train:]
        self.dev_data.loc[self.dev_data.study_id.isin(valid_sub_id), "split"]="validation"
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']

        self.train_set = MammoDataset(self.train_data, data_dir, self.image_size, augmentation=True)
        self.val_set = MammoDataset(self.val_data, data_dir, self.image_size, augmentation=False)
        self.test_set = MammoDataset(self.test_data, data_dir, self.image_size, augmentation=False)

        train_labels = self.train_set.get_labels()        
        train_class_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])

        val_labels = self.val_set.get_labels()        
        val_class_count = np.array([len(np.where(val_labels == t)[0]) for t in np.unique(val_labels)])

        test_labels = self.test_set.get_labels()        
        test_class_count = np.array([len(np.where(test_labels == t)[0]) for t in np.unique(test_labels)])

        print('samples (train): ',len(self.train_set))
        print('samples (val):   ',len(self.val_set))
        print('samples (test):  ',len(self.test_set))
        print('pos/neg (train): {}/{}'.format(train_class_count[1], train_class_count[0]))
        print('pos/neg (val):   {}/{}'.format(val_class_count[1], val_class_count[0]))
        print('pos/neg (test):  {}/{}'.format(test_class_count[1], test_class_count[0]))
        print('pos (train):     {:0.2f}%'.format(train_class_count[1]/len(train_labels)*100.0))
        print('pos (val):       {:0.2f}%'.format(val_class_count[1]/len(val_labels)*100.0))
        print('pos (test):      {:0.2f}%'.format(test_class_count[1]/len(test_labels)*100.0))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, sampler=ImbalancedDatasetSampler(self.train_set), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class MammoNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 2        
        self.predictions = []
        self.targets = []
        self.study_ids = []
        self.image_ids = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []

        # Default model is an ImageNet pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def process_batch(self, batch):
        img, lab = batch['image'], batch['label']
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = F.cross_entropy(out, lab)        
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, batch_size=batch_size)        
        batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=batch_size)                        
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss, batch_size=batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('val_auc', auc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []
        self.study_ids = []
        self.image_ids = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())
        self.study_ids.append(batch['study_id'])
        self.image_ids.append(batch['image_id'])


def save_predictions(model, output_fname):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)
    std_ids = [id for sublist in model.study_ids for id in sublist]
    img_ids = [id for sublist in model.image_ids for id in sublist]

    auc = auroc(prds, trgs, num_classes=2, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, 2)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df['study_id'] = std_ids
    df['image_id'] = img_ids
    df.to_csv(output_fname, index=False)


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = MammoDataModule(data_dir=data_dir,
                              image_size=image_size,
                              val_percent=val_percent,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    model = MammoNet()

    # Create output directory
    output_base_dir = 'output'
    output_name = 'resnet18'
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('=============================================================')
    print('Training...')

    # train
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator=hparams.dev,
        devices=1,
        logger=TensorBoardLogger(output_base_dir, name=output_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = MammoNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    print('=============================================================')
    print('Testing...')

    trainer.test(model=model, datamodule=data)
    save_predictions(model=model, output_fname=os.path.join(output_dir, 'predictions.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dev', default='gpu')
    args = parser.parse_args()

    main(args)
