import torch


class Trainer:
    """Loop de treino flexível para modelos PyTorch com interface TFB-like."""

    def __init__(self, model, optimizer, device=None, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_batch(self, batch):
        if len(batch) == 2:
            batch_x, batch_y = batch
            batch_candle = None
        elif len(batch) == 3:
            batch_x, batch_y, batch_candle = batch
            batch_candle = batch_candle.to(self.device)
        else:
            raise ValueError(f"Batch inesperado com {len(batch)} elementos")

        return batch_x.to(self.device), batch_y.to(self.device), batch_candle

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            batch_x, batch_y, batch_candle = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            forward_kwargs = {}
            if batch_candle is not None:
                forward_kwargs["candle_x"] = batch_candle

            _, loss = self.model(batch_x, batch_y, return_loss=True, **forward_kwargs)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / max(1, len(dataloader))

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        for batch in dataloader:
            batch_x, batch_y, batch_candle = self._prepare_batch(batch)
            forward_kwargs = {}
            if batch_candle is not None:
                forward_kwargs["candle_x"] = batch_candle

            _, loss = self.model(batch_x, batch_y, return_loss=True, **forward_kwargs)
            running_loss += loss.item()

        return running_loss / max(1, len(dataloader))
