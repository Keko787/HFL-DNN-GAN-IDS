# CAN GAN

A Hierarchical Federated Learning and GAN-based Network Intrusion Detection System for private and IoT networks.

## Table of Contents

- [Team](#team)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Federated Training (Host)](#federated-training-host)
  - [Localized & Federated Training (Client)](#localized--federated-training-client)
- [Architecture](#architecture)
- [Models](#models)
- [License](#license)

---

## Team

**Faculty Advisors (2023–2025)**

- Dr. Chenqi Qu


---

## Prerequisites

- Ubuntu 22.04 LTS with CUDA 12 drivers (P100, M40 support)
- Python 3.8+

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Keko787/HFL-DNN-GAN-IDS.git
cd HFL-DNN-GAN-IDS

# [Option 1] AERPAW node setup
python3 AppSetup/AERPAW_node_Setup.py

# [Option 2] Chameleon node setup
python3 AppSetup/Chameleon_node_Setup.py

```

---

## Datasets

1. Download **CIC IoT2023** from [CIC website](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
2. Upload `CICIoT2023.zip` to `$HOME/datasets/`, then:

```bash
unzip $HOME/datasets/CICIoT2023.zip -d $HOME/datasets/CICIOT2023
```
Then run the upgrade to tensorflow-addons
```bash
pip install --upgrade tensorflow-addons==0.23.0
```
---

### Localized & Federated Training (Client)

```bash
python3 App/TrainingApp/Client/TrainingClient.py --help
# Default uses CICIOT2023 dataset; use --dataset IOTBOTNET for IoTBotnet.
```

---

## Models

- **Discriminator**: Multi-class classifier (real vs. synthetic).
- **Generator**: GAN-based traffic synthesizer.
- **GAN**: Combined model for adversarial data generation and classification.

---

## License

This project is licensed under the [MIT License](LICENSE).

