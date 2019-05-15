def run():

    from pathlib import Path
    import torch
    from utils import get_yaml_config, get_dataloader
    from models import CycleGan

    ROOT = Path(".")
    DATA = ROOT / "data"
    CONFIG = get_yaml_config(ROOT / "config.yml", verbose=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader_x, testloader_x = get_dataloader(
        "summer", DATA / CONFIG.image_path, CONFIG.image_size, CONFIG.batch_size
    )
    trainloader_y, testloader_y = get_dataloader(
        "winter", DATA / CONFIG.image_path, CONFIG.image_size, CONFIG.batch_size
    )

    g = CycleGan(CONFIG, trainloader_x, trainloader_y, testloader_x, testloader_y, DEVICE)
    g.train()


if __name__ == "__main__":
    run()
