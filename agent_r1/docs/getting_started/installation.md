### Environment Setup

We used GPU Base Image 22.04 on Lambda Labs with 4xA100s 80GB SXM

**Clone the repository**
```bash
git clone https://github.com/alphaXiv/Agent-R1.git
cd Agent-R1
```

**Install `docker image for verl+vllm rollout`**
```
sudo docker pull hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0

```

Rest of the setup and procedure for training is given in [quickstart.md](./quickstart.md)
