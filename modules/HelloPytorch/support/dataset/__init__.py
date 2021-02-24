from torch.utils.data.dataloader import Dataset


class DataHolder(Dataset):
    import numpy as np
    from torch.utils.data.dataset import T_co

    def __init__(self, x: np.ndarray, y: np.ndarray, x_transform=None, y_transform=None) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> T_co:
        return (
            self.x[index] if self.x_transform is None else self.x_transform(self.x[index]),
            self.y[index] if self.y_transform is None else self.y_transform(self.y[index])
        )