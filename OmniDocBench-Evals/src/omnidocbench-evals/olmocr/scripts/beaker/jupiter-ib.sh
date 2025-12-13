set -ex
export NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=ib NCCL_IB_HCA="^=mlx5_bond_0"