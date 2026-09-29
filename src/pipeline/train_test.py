"""Train and evaluate the Xception baseline for one validation fold."""

import random
import warnings
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from src.utilities import calculate_metrics, PROJECT_ROOT
from src.pipeline.preprocessing import get_preprocessing_transform


class ImagePathDataset(Dataset):
	def __init__(self, image_paths, labels, transform):
		self.image_paths = image_paths
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image = read_image(
			str(self.image_paths[index]),
			mode=ImageReadMode.RGB,
		)
		image_tensor = self.transform(image)
		return image_tensor, int(self.labels[index])
		

def build_model(model_cfg, pretrained=None):
	backbone = timm.create_model(
		model_cfg['backbone'],
		pretrained=bool(model_cfg['pretrained'] if pretrained is None else pretrained),
		num_classes=0,
	)
	classifier_cfg = model_cfg.get('classifier_head', {})
	classifier_layers = []
	input_features = backbone.num_features
	for layer_cfg in classifier_cfg.get('layers', []):
		units = int(layer_cfg['units'])
		classifier_layers.append(nn.Linear(input_features, units))
		activation = layer_cfg.get('activation', '').lower()
		if activation == 'relu':
			classifier_layers.append(nn.ReLU())
		dropout = float(classifier_cfg.get('dropout', layer_cfg.get('dropout', 0)))
		if dropout:
			classifier_layers.append(nn.Dropout(dropout))
		input_features = units
	classifier_layers.append(nn.Linear(input_features, 2))
	model = nn.Sequential(backbone, nn.Sequential(*classifier_layers))

	freeze_backbone = bool(model_cfg.get('freeze_backbone', False))
	for parameter in backbone.parameters():
		parameter.requires_grad = not freeze_backbone

	return model, backbone


def train_and_evaluate(fold, X_train, y_train, X_val, y_val, cfg):
	seed = int(cfg.get('seed', 42))
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	transform = get_preprocessing_transform()
	train_dataset = ImagePathDataset(X_train, y_train, transform=transform)
	val_dataset = ImagePathDataset(X_val, y_val, transform=transform)
	train_loader = DataLoader(
		train_dataset,
		**cfg['data'],
		shuffle = True
	)
	val_loader = DataLoader(
		val_dataset,
		**cfg['data']
	)

	### MODEL ###
	model_cfg = cfg['model']
	model, backbone = build_model(model_cfg)
	freeze_backbone = bool(model_cfg.get('freeze_backbone', False))

	if model_cfg.get('facial_coordinates', {}).get('enabled', False):
		warnings.warn(
			'facial_coordinates.enabled is true, but this dataset currently returns '
			'only images and labels; coordinate fusion is not applied.',
			RuntimeWarning,
		)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	
	optimization_cfg = cfg['optimization']
	criterion = getattr(nn, optimization_cfg['loss_function'])()
	optimizer_cls = getattr(torch.optim, optimization_cfg['optimizer'])
	optimizer = optimizer_cls(
		(parameter for parameter in model.parameters() if parameter.requires_grad),
		lr=float(optimization_cfg['learning_rate']),
		weight_decay=optimization_cfg['weight_decay'],
	)

	scheduler_cfg = optimization_cfg['scheduler']
	scheduler = None
	if scheduler_cfg:
		scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cfg['type'])
		scheduler_options = {
			key: value for key, value in scheduler_cfg.items() if key != 'type'
		}
		scheduler = scheduler_cls(optimizer, **scheduler_options)

	early_stopping_cfg = optimization_cfg.get('early_stopping', {})
	early_stopping_patience = int(early_stopping_cfg.get('patience', 0))
	best_val_loss = float('inf')
	best_metrics = None
	epochs_without_improvement = 0
	max_epochs = int(optimization_cfg['max_epochs'])
	checkpoint_dir = PROJECT_ROOT / cfg['checkpoint_dir']
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = checkpoint_dir / f"{cfg['experiment_name']}_{fold}.pt"

	for epoch in range(1, max_epochs + 1):
		model.train()
		if freeze_backbone:
			backbone.eval()
		for batch_index, (X_batch, y_batch) in enumerate(train_loader, start=1):
			X_batch = X_batch.to(device)
			y_batch = y_batch.long().to(device)
			logits = model(X_batch)
			loss = criterion(logits, y_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if batch_index % 25 == 0 or batch_index == 1:
				print(f"epoch {epoch}/{max_epochs} train {batch_index}/{len(train_loader)}")

		model.eval()
		total_loss = 0.0
		validation_labels = []
		validation_predictions = []
		positive_scores = []
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch = X_batch.to(device)
				y_batch = y_batch.long().to(device)
				logits = model(X_batch)
				loss = criterion(logits, y_batch)
				total_loss += loss.item() * y_batch.size(0)
				validation_labels.append(y_batch.cpu().numpy())
				validation_predictions.append(logits.argmax(dim=1).cpu().numpy())
				positive_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

		y_true = np.concatenate(validation_labels).astype(np.int64)
		y_pred = np.concatenate(validation_predictions).astype(np.int64)
		y_score = np.concatenate(positive_scores)
		val_loss = total_loss / len(val_dataset)
		metrics = calculate_metrics(
			y_true=y_true,
			y_pred=y_pred,
			y_score=y_score,
			val_loss=val_loss,
		)
		if scheduler is not None:
			scheduler.step(val_loss)
		print(
			f"{fold} epoch {epoch}/{max_epochs}: acc={metrics['acc']:.4f}, "
			f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
			f"val_loss={metrics['val_loss']:.4f}"
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_metrics = metrics
			epochs_without_improvement = 0
			torch.save({
				'model_state_dict': {
					key: value.detach().cpu()
					for key, value in model.state_dict().items()
				},
				'model_config': model_cfg,
				'experiment_name': cfg['experiment_name'],
				'fold': fold,
				'epoch': epoch,
				'metrics': metrics,
			}, checkpoint_path)
			print(f"Saved best model to {checkpoint_path}")
		else:
			epochs_without_improvement += 1
			if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
				print(f"Early stopping after epoch {epoch}.")
				break

	return best_metrics