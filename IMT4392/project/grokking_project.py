{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "%pylab is deprecated, use %matplotlib inline and import the required libraries.\n",
      "Populating the interactive namespace from numpy and matplotlib\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torch'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Input \u001b[0;32mIn [1]\u001b[0m, in \u001b[0;36m<cell line: 3>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m get_ipython()\u001b[38;5;241m.\u001b[39mrun_line_magic(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpylab\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124minline\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[0;32m----> 3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnn\u001b[39;00m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mfunctional\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mF\u001b[39;00m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'torch'"
     ]
    }
   ],
   "source": [
    "%pylab inline\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.utils.data.dataloader as dataloader\n",
    "import torch.optim as optim\n",
    "\n",
    "from torch.utils.data import TensorDataset\n",
    "from torch.autograd import Variable\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "from torchvision.datasets import MNIST\n",
    "\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import time\n",
    "from datetime import timedelta\n",
    "import math\n",
    "import pdb\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_epochs = 5\n",
    "num_classes = 10\n",
    "batch_size = 15\n",
    "learning_rate = 0.001"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dataset = torchvision.datasets.MNIST(root='./data/',\n",
    "                                           train=True, \n",
    "                                           transform=transforms.ToTensor(),\n",
    "                                           download=True)\n",
    "\n",
    "test_dataset = torchvision.datasets.MNIST(root='./data/',\n",
    "                                          train=False, \n",
    "                                          transform=transforms.ToTensor())\n",
    "\n",
    "# Data loader\n",
    "train_loader = torch.utils.data.DataLoader(dataset=train_dataset,\n",
    "                                           batch_size=batch_size, \n",
    "                                           shuffle=True)\n",
    "\n",
    "test_loader = torch.utils.data.DataLoader(dataset=test_dataset,\n",
    "                                          batch_size=batch_size, \n",
    "                                          shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Data\n",
      " - Number of digit: (60000, 28, 28)\n"
     ]
    }
   ],
   "source": [
    "print('Training Data')\n",
    "print(' - Number of digit:', train_dataset.data.cpu().numpy().shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Data\n",
      " - Number of digit: (10000, 28, 28)\n"
     ]
    }
   ],
   "source": [
    "print('Test Data')\n",
    "print(' - Number of digit:', test_dataset.data.cpu().numpy().shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![title](network_flowchart.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Layer 1\n",
    "filter_size1 = 5\n",
    "num_filter1  =16\n",
    "# Layer 2\n",
    "filter_size2 = 5\n",
    "num_filter2  =36\n",
    "#Fully connected\n",
    "num_neuron =  128\n",
    "num_channels = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We know that MNIST images are 28 pixels in each dimension.\n",
    "img_size = 28\n",
    "\n",
    "# Images are stored in one-dimensional arrays of this length.\n",
    "img_size_flat = img_size * img_size\n",
    "\n",
    "# Tuple with height and width of images used to reshape arrays.\n",
    "img_shape = (img_size, img_size)\n",
    "\n",
    "# Number of colour channels for the images: 1 channel for gray-scale.\n",
    "num_channels = 1\n",
    "\n",
    "# Number of classes, one class for each of 10 digits.\n",
    "num_classes = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_images(images, cls_true, cls_pred=None):\n",
    "    assert len(images) == len(cls_true) == 9\n",
    "    \n",
    "    # Create figure with 3x3 sub-plots.\n",
    "    fig, axes = plt.subplots(3, 3)\n",
    "    fig.subplots_adjust(hspace=0.3, wspace=0.3)\n",
    "\n",
    "    for i, ax in enumerate(axes.flat):\n",
    "        # Plot image.\n",
    "        ax.imshow(images[i].reshape(img_shape), cmap='binary')\n",
    "\n",
    "        # Show true and predicted classes.\n",
    "        if cls_pred is None:\n",
    "            xlabel = \"True: {0}\".format(cls_true[i])\n",
    "        else:\n",
    "            xlabel = \"True: {0}, Pred: {1}\".format(cls_true[i], cls_pred[i])\n",
    "\n",
    "        # Show the classes as the label on the x-axis.\n",
    "        ax.set_xlabel(xlabel)\n",
    "        \n",
    "        # Remove ticks from the plot.\n",
    "        ax.set_xticks([])\n",
    "        ax.set_yticks([])\n",
    "    \n",
    "    # Ensure the plot is shown correctly with multiple plots\n",
    "    # in a single Notebook cell.\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_kernels(tensor, num_cols=6):\n",
    "    if not tensor.ndim==4:\n",
    "        raise Exception(\"assumes a 4D tensor\")\n",
    "    if not tensor.shape[-1]==3:\n",
    "        raise Exception(\"last dim needs to be 3 to plot\")\n",
    "    num_kernels = tensor.shape[0]\n",
    "    num_rows = 1+ num_kernels // num_cols\n",
    "    fig = plt.figure(figsize=(num_cols,num_rows))\n",
    "    for i in range(tensor.shape[0]):\n",
    "        ax1 = fig.add_subplot(num_rows,num_cols,i+1)\n",
    "        ax1.imshow(tensor[i])\n",
    "        ax1.axis('off')\n",
    "        ax1.set_xticklabels([])\n",
    "        ax1.set_yticklabels([])\n",
    "\n",
    "    plt.subplots_adjust(wspace=0.1, hspace=0.1)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "examples = enumerate(test_loader)\n",
    "batch_idx, (example_data, example_targets) = next(examples)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([15, 1, 28, 28])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "example_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 9 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_images(example_data[0:9].permute(0,2,3,1).cpu().numpy(),example_targets[0:9].cpu().numpy())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([15, 28, 28, 1])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "example_data.permute(0,2,3,1).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, num_filter1, kernel_size=filter_size1,padding=2)\n",
    "        self.conv2 = nn.Conv2d(num_filter1, num_filter2, kernel_size=filter_size2,padding=2)\n",
    "        self.relu  = nn.ReLU()\n",
    "        self.fc1 = nn.Linear(1764, num_neuron)\n",
    "        self.fc2 = nn.Linear(num_neuron, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x=self.conv1(x)\n",
    "        x= F.max_pool2d(x,2)\n",
    "        w1=self.relu(x)\n",
    "        x = self.conv2(w1)\n",
    "        x=self.relu(x)\n",
    "        w2=F.max_pool2d(x,2)\n",
    "        #pdb.set_trace()\n",
    "        x = w2.view(-1, 1764)\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.fc2(x)\n",
    "        return (x,w1,w2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "momentum=0.9\n",
    "model = Net().to(device)\n",
    "optimizer = optim.SGD(model.parameters(), lr=learning_rate,\n",
    "                      momentum=momentum)\n",
    "criterion = nn.CrossEntropyLoss()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_image(image):\n",
    "    plt.imshow(image.reshape(img_shape),\n",
    "               interpolation='nearest',\n",
    "               cmap='binary')\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAa00lEQVR4nO3df2zU9R3H8deB9ARtr6ulvd4orICWKVAzlK5DEUcDrRkRJYu//gBDIGJxw85puijIWFIHiyM6Bst+0JmIOjeBSRYSLbbMrWUDYYS4dbSpgqEtk427UqQw+tkfxBsH5cf3uOu7V56P5JvQu++n9/a7b/rcl7t+8TnnnAAA6GODrAcAAFydCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBxjfUA5+rp6dGhQ4eUnp4un89nPQ4AwCPnnDo7OxUKhTRo0IWvc/pdgA4dOqT8/HzrMQAAV+jgwYMaMWLEBZ/vdwFKT0+XdGbwjIwM42kAAF5FIhHl5+dHf55fSNICtGbNGq1atUrt7e0qKirSyy+/rMmTJ19y3ed/7ZaRkUGAACCFXeptlKR8COGNN95QZWWlli1bpg8++EBFRUWaOXOmDh8+nIyXAwCkoKQE6MUXX9SCBQv06KOP6uabb9a6des0bNgw/epXv0rGywEAUlDCA3Ty5Ent2rVLpaWl/3+RQYNUWlqqhoaG8/bv7u5WJBKJ2QAAA1/CA/Tpp5/q9OnTys3NjXk8NzdX7e3t5+1fXV2tQCAQ3fgEHABcHcx/EbWqqkrhcDi6HTx40HokAEAfSPin4LKzszV48GB1dHTEPN7R0aFgMHje/n6/X36/P9FjAAD6uYRfAaWlpWnSpEmqra2NPtbT06Pa2lqVlJQk+uUAACkqKb8HVFlZqblz5+q2227T5MmTtXr1anV1denRRx9NxssBAFJQUgL0wAMP6F//+peWLl2q9vZ23Xrrrdq6det5H0wAAFy9fM45Zz3E2SKRiAKBgMLhMHdCAIAUdLk/x80/BQcAuDoRIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATCQ/Q888/L5/PF7ONGzcu0S8DAEhx1yTjm95yyy169913//8i1yTlZQAAKSwpZbjmmmsUDAaT8a0BAANEUt4D2r9/v0KhkEaPHq1HHnlEBw4cuOC+3d3dikQiMRsAYOBLeICKi4tVU1OjrVu3au3atWptbdWdd96pzs7OXvevrq5WIBCIbvn5+YkeCQDQD/mccy6ZL3D06FGNGjVKL774oubPn3/e893d3eru7o5+HYlElJ+fr3A4rIyMjGSOBgBIgkgkokAgcMmf40n/dEBmZqZuuukmNTc39/q83++X3+9P9hgAgH4m6b8HdOzYMbW0tCgvLy/ZLwUASCEJD9BTTz2l+vp6ffTRR/rzn/+s++67T4MHD9ZDDz2U6JcCAKSwhP8V3CeffKKHHnpIR44c0fDhw3XHHXeosbFRw4cPT/RLAQBSWMID9Prrryf6WwIABiDuBQcAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmEj6P0iHvvXb3/7W85qf//zncb1WKBTyvObaa6/1vOaRRx7xvCYYDHpeI0ljx46Nax0A77gCAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAmfc85ZD3G2SCSiQCCgcDisjIwM63FSTkFBgec1H330UeIHMRbvuXPzzTcneBIkWn5+vuc1Tz/9dFyvddttt8W17mp3uT/HuQICAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAExcYz0AEusXv/iF5zV/+9vf4nqteG7c+eGHH3pes3v3bs9r6urqPK+RpMbGRs9rRo4c6XnNgQMHPK/pS0OGDPG8Jjs72/OatrY2z2vi+d8onhuYStyMNNm4AgIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATHAz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dXVF93nyySf19ttv680331R9fb0OHTqk+++/P+GDAwBSm6cPIWzdujXm65qaGuXk5GjXrl2aOnWqwuGwfvnLX2rDhg36+te/Lklav369vvzlL6uxsVFf/epXEzc5ACClXdF7QOFwWJKUlZUlSdq1a5dOnTql0tLS6D7jxo3TyJEj1dDQ0Ov36O7uViQSidkAAANf3AHq6enRkiVLNGXKFI0fP16S1N7errS0NGVmZsbsm5ubq/b29l6/T3V1tQKBQHTLz8+PdyQAQAqJO0AVFRXat2+fXn/99SsaoKqqSuFwOLodPHjwir4fACA1xPWLqIsXL9aWLVu0fft2jRgxIvp4MBjUyZMndfTo0ZiroI6ODgWDwV6/l9/vl9/vj2cMAEAK83QF5JzT4sWLtXHjRm3btk0FBQUxz0+aNElDhgxRbW1t9LGmpiYdOHBAJSUliZkYADAgeLoCqqio0IYNG7R582alp6dH39cJBAIaOnSoAoGA5s+fr8rKSmVlZSkjI0NPPPGESkpK+AQcACCGpwCtXbtWkjRt2rSYx9evX6958+ZJkn784x9r0KBBmjNnjrq7uzVz5kz99Kc/TciwAICBw+ecc9ZDnC0SiSgQCCgcDisjI8N6HFxl/vnPf3peU1hYmIRJzvf73//e85pZs2YlYRLg4i735zj3ggMAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAICJuP5FVKC/+/jjj+NaN2PGjARP0rsf/ehHntd84xvfSMIkgB2ugAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAE9yMFAPSz372s7jWxXsTU6/uuusuz2t8Pl8SJgHscAUEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJjgZqTo9/74xz96XvOTn/wkCZMASCSugAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAE9yMFP3e+++/73lNZ2dnEibp3dixYz2vuf7665MwCZBauAICAJggQAAAE54CVF1drdtvv13p6enKycnR7Nmz1dTUFLPPtGnT5PP5YrbHHnss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ipnPzwHOj/ONHj1a2dnZA/L8WLx4sbZs2aL33nsv5p9vCQaDOnnypI4ePRqz/0A9Hy50HHpTXFwsSf3qfOj3AUpLS9OkSZNUW1sbfaynp0e1tbUqKSkxnMzesWPH1NLSory8POtRzBQUFCgYDMacH5FIRDt27Ljqz49PPvlER44cGVDnh3NOixcv1saNG7Vt2zYVFBTEPD9p0iQNGTIk5nxoamrSgQMHBtT5cKnj0Js9e/ZIUv86H6w/BXE5Xn/9def3+11NTY378MMP3cKFC11mZqZrb2+3Hq1Pfec733F1dXWutbXV/elPf3KlpaUuOzvbHT582Hq0pOrs7HS7d+92u3fvdpLciy++6Hbv3u0+/vhj55xzL7zwgsvMzHSbN292e/fudffee68rKChwn332mfHkiXWx49DZ2emeeuop19DQ4FpbW927777rvvKVr7gbb7zRnThxwnr0hFm0aJELBAKurq7OtbW1Rbfjx49H93nsscfcyJEj3bZt29zOnTtdSUmJKykpMZw68S51HJqbm933v/99t3PnTtfa2uo2b97sRo8e7aZOnWo8eayUCJBzzr388stu5MiRLi0tzU2ePNk1NjZaj9TnHnjgAZeXl+fS0tLcF7/4RffAAw+45uZm67GS7r333nOSztvmzp3rnDvzUeznnnvO5ebmOr/f76ZPn+6amppsh06Cix2H48ePuxkzZrjhw4e7IUOGuFGjRrkFCxYMuP+T1tt/vyS3fv366D6fffaZe/zxx90XvvAFN2zYMHffffe5trY2u6GT4FLH4cCBA27q1KkuKyvL+f1+N3bsWPfd737XhcNh28HPwT/HAAAw0e/fAwIADEwECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgIn/AXUYjuKM3UN2AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "image1 = example_data[0].permute(1,2,0).cpu().numpy()\n",
    "plot_image(image1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "outputs = model(example_data[0].view(1,1,28,28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([1, 16, 14, 14])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "outputs[1].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_conv_weights(weights, input_channel=0):\n",
    "    w=weights\n",
    "    w_min = np.min(w)\n",
    "    w_max = np.max(w)\n",
    "    #shape =[filter_size,filter_size,num_channels,num_filters]\n",
    "    # Number of filters used in the conv. layer.\n",
    "    num_filters = w.shape[3]\n",
    "    print(num_filters)\n",
    "    # Number of grids to plot.\n",
    "    # Create figure with a grid of sub-plots.\n",
    "    num_grids = math.ceil(math.sqrt(num_filters))\n",
    "    fig, axes = plt.subplots(num_grids, num_grids)\n",
    "\n",
    "    # Plot all the filter-weights.\n",
    "    for i, ax in enumerate(axes.flat):\n",
    "        # Only plot the valid filter-weights.\n",
    "        if i<num_filters:\n",
    "            # Get the weights for the i'th filter of the input channel.\n",
    "            # See new_conv_layer() for details on the format\n",
    "            # of this 4-dim tensor.\n",
    "            img = w[:, :, input_channel, i]\n",
    "\n",
    "            # Plot image.\n",
    "            ax.imshow(img, vmin=w_min, vmax=w_max,\n",
    "                      interpolation='nearest', cmap='seismic')\n",
    "        \n",
    "        # Remove ticks from the plot.\n",
    "        ax.set_xticks([])\n",
    "        ax.set_yticks([])\n",
    "    \n",
    "    # Ensure the plot is shown correctly with multiple plots\n",
    "    # in a single Notebook cell.\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "16\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAe0AAAGKCAYAAAAhRRkZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAd40lEQVR4nO3df5TdZX0n8GcCiUEcKCsBk5OI8ZCFw5olWuKBFXtEitVlQeuiiyJVtrLSioewW1aUwqqUgqYe9GhbXFiQHnXbqtWV7ZGKFFbqBjcV8GAxnFAQJ50ghA0wQMZMmLt/JOzg+X4+ydw798c8M6/XXznveeb5Prnz3PuZb/K5zx1qtVqtAgDMegsGvQAAYHoUbQCohKINAJVQtAGgEoo2AFRC0QaASijaAFAJRRsAKrF/p984OTlZRkdHy/DwcBkaGurmmuaFVqtVxsbGyrJly8qCBXX+7mQPzJx9gD1AO3ug46I9OjpaVqxY0em3s8fIyEhZvnz5oJfREXuge+wD7AGmswc6LtrDw8N7/nTyTKaZx3aVUm59weNYn6m1n1dKedEgl1KxX5RSrpkj+8BrQWfm0mvBH5dSDhjkUiq1o5TywWntgY6fYVP/BLJ/KWVhp9PMezX/U9LU2l9UFO2ZmRv7wGvBTMyNPXBAKeXFg1xK1aazB+r8DxQAmIcUbQCohKINAJVQtAGgEoo2AFRC0QaASijaAFAJRRsAKtHn44teE2QnJWM/3cuFMDCXNZK1aw8MR27c+Ewyx/eT/K4g2z6tVQH9tibIdiZj7+vhOuriThsAKqFoA0AlFG0AqISiDQCV6Gsj2oUXfryRffjD8dgTTnhDmJ9xRjz+ppua2U9/Go8dH388/kJ5Lsl/O8lp10UXNZvO9k924TXXxA1qRx75pjB/+ulmfs89015aKaWUJ56I87POuj/5juizbz/R3kXZ48gkf6Cvq6Bf9mskq1atDkceemicL14cz3zbbdFrefYcfirJm+vbLasTdwbZ0mRs59xpA0AlFG0AqISiDQCVULQBoBKKNgBUoq/d41dffVqQtTfH+vWnJ185KMgWJWPj/KSTTg3z227b57KYpvXro879+HfHK6/8rS5c8bAkvyZMV69e2NbsJ53U7HC3XzrVrS7xqKX4gGRs9LpRSikPd2kt5L7TSDZvjp88mzc/kcwR14PVq49qZIsXHxOOfdWr4pmzd7X8+MdxvmFDdOzytnjwDLjTBoBKKNoAUAlFGwAqoWgDQCUUbQCoRF+7x2NxR18pW5L8R0l+RJDFbYGtc74a5v/xV3SP917UTZl1+a9J8pcn+WSQ7QhHnnBC3CW+YUPzHQ67/UGY3nbbR5PxDM74NLNSStney4WwV80O70svPTkc+cUvxjOcd16cR587kXWJv/nNcZ51if/938d5KY8GWffvi91pA0AlFG0AqISiDQCVULQBoBKKNgBUYhZ0j9/X5vinkrx5VvC73nVRPPSqXw3jqw/POoez84mztZC7vo2x+yV51KVZSimHNpJVq9aFI487Lp5h8eKbwnzNmnj81VdHf5/o7GvglzXfSXL55X/R1gyXXHJI8pXljeSkk+J3Kl1wQfyOpLPPPjbMH3kkW81zQaZ7HADmLUUbACqhaANAJRRtAKjELGhE650TT4zzocPjIyxzS5NcI1pvRY0dpeSP++pGcv758cgLLnggzD/72SOT8VmTYnYML9B72TG0BzaS225rNiuXUsopp8QNZwc2pyillLJ1a9YI225d6Yw7bQCohKINAJVQtAGgEoo2AFRC0QaASsyZ7vGvf715/OSRcSNwKeW3kjw7fvL+DlZE78Q/p7PPPr2RHX10PMO558ab4yc/ya551DTWBfRXdtTxqiC7MRz5trfFx11/8INbk7lvT/L+3AO70waASijaAFAJRRsAKqFoA0AlFG0AqESF3eNXh+nb3/xsIxs6MB6bdxxSg6Ghj4f54qCp/Dd+44pw7Je/fEmYn3WWM8ahHr+W5M2Dw1etirvE77yz3WsO9l7XnTYAVELRBoBKKNoAUAlFGwAqoWgDQCWq6x5vPffKMP/Dq6LfP7K2wEOTfFsnS6Jn4s7Qb34zHv17v9fMLrww7hL/2Meya+oSh9nniCRfmOQPNpI1a1aHI7/61eya39vXogbCnTYAVELRBoBKKNoAUAlFGwAqoWgDQCVmbff4aafdFOajj8TjL7kkOzM68lT7C6LvVq48Ncx/8pN4/E9/2szuuSceu3nzmclVsy5VYHDidw2V8r/D9CMfab52fPe78Qzj49/vcE2D4U4bACqhaANAJRRtAKiEog0AlZi1jWiZ88/vxiw7uzEJPfaSl8T5F78Y5xMT9zWy2267NpldwxnU4+EkPypM/3Ddo43syiubrw+7/byzJQ2IO20AqISiDQCVULQBoBKKNgBUQtEGgErM2u7xm25q51jSUko5Psiy40qzLkJmk3vv3ZJ85dAkvz7IHkjGHtP+goAB+VaYnnPO5WH+zvBdRlk90D0OAPSAog0AlVC0AaASijYAVKLjRrRWq7XnT7u6tJSZGg+yXyRjJ3q5kGna/bhNPY71mVp79jjP1FiSL0ryaB3Zz7pXa27X7nXMjX0wW14LajOXXgt29OgK8fN15864uWwifNpnryfPdrSi7tr9uE1nDwy1OtwpW7ZsKStWrOjkW3mBkZGRsnz58kEvoyP2QPfYB9gDTGcPdFy0Jycny+joaBkeHi5DQ0MdLXA+a7VaZWxsrCxbtqwsWFDn/1LYAzNnH2AP0M4e6LhoAwD9VeevdQAwDynaAFAJRRsAKqFoA0AlFG0AqISiDQCVULQBoBKKNgBUQtEGgEoo2gBQCUUbACqhaANAJTr+PG2f6jIzPtmHUuwD7AHa2wMdF+3R0VGfn9oFNX+Grj3QPfYB9gDT2QMdF+3h4eE9fzp5JtPMY7tKKbe+4HGsz9TazymlLBrkUiq2s5RywxzZB14LOjOXXgveVkpZOMCV1GqilPLNae2Bjp9hU/8Esn/xQ+pczf+UNLX2RaWUFw1yKdWbG/vAa8FMzI09sLDYA52bzh6o8z9QAGAeUrQBoBKKNgBUQtEGgEoo2gBQCUUbACqhaANAJRRtAKhEn48v+s0gOzkcuXTpEWG+desDydzXBNn901oVg7Y2yQ9K8u8l+UQX1sLgHJPk+yX5vb1aCH1xRpAdmoz9qyR/tEtrqYc7bQCohKINAJVQtAGgEoo2AFSiz41ozUahJ5+MG84Oet/b4ynOPDPOnw6aWPb/l/HYJ56I87GxMP6/v3NJmL/0pVEjzEfjudnjwSDLfnd8JskPT/KocS1rYsocmORPJfl9QZatj71bHKZXXPHxMH/66XiW229vZg8k/au7dsX5+94X59lLxw03rA/SrGGS3aLnVPac//UwPeSQU8N8+/botfkv21hHKaUsTfKswXlbkGXNtJ1zpw0AlVC0AaASijYAVELRBoBKKNoAUIk+d4+/vJEcnjTajo9Hx5KWcuErDgvz889vZq94RTz3gj/94/gLS+NuwX/2+U+E+YoVlzWykZF4ap737xrJq1/9nnDknXfGMyzaNhp/IWgn3vmKfx4O3T/Z+TffHOf/+rj4uMShw6NO0q/Fk7APd4XpJZe8LRn/XM9WcvXV68L80kvjY5dLifaw7vG9i17LXxmOHBo6Ksy3b/+TZO6VQfaqZOyaJD8yybPu8a1Bdn0ytnPutAGgEoo2AFRC0QaASijaAFAJRRsAKtHn7vE3NZJPfjIeecEFO8P87/4uHv/tbzezJ5+Mx37pSx8M8zf+p1eH+Z+ce3eYj4ycFl+AvXi4kdx993fCkcce29wvpZRyyCHLwjw6lj47K/plL4vz/3Bmcg7xe94fxldd9a1GdvHFusc7c0aYrlz53jDP3gGwefN4G9eM994VV8Rd4h/d/1NhfnlZ0cY12S06C/xd4chW69AwP+GE3w3zDRseb2SHHPLScOxnPhPG5cor4/yZZ4LPuSilPPJIM5+Y0D0OAPOWog0AlVC0AaASijYAVELRBoBK9Ll7/G2N5IILsrGLw3TjxuXJ+Ch/bTjyjef9dpj/7ebNYb5mTXJJOhCd4f3X4chNm+K8lIPCdMOG5tn2pWSdxPG51e9//7owf+CPml3ipZRya3DmPZ2Ku+4feqj/3fhPPHF6/IUzfz35jq/0bC1zV/QOoRuTsXG+YcMh0557+/Z/E4687rr4Z71pU/zhB+ecc3yY33DDtclausudNgBUQtEGgEoo2gBQCUUbACrR50a0dmQNRA8kebM5ae3a14cj/3Zj3HD2xmTmodc5rnR2SY4aLT8OsgfDkWeffVOYL3g6nvvpp+Pmt1tu+UKyFmr2qasmw3xov9XJd9zfu8WwF9uT/Kggi+vBHXfEjclr1/63MN+0KVvLj7IvdJU7bQCohKINAJVQtAGgEoo2AFRC0QaASszi7vFMfLxpKQc0kps3DoUj/08yw2vXtuIvbNQ9XocdQfZr4ciLL45n+MFP4i7xs8/OrnnfPlfF7PUP/xC/i6CMP5t8R7oRGIisHpwcZBPJ2MvauuKGDeuTr2Sd7N3lThsAKqFoA0AlFG0AqISiDQCVULQBoBIVdo+/KUyffPLcRnbzwR8Jx75zdXx+8Ft0iVduayN5xzv+azhy1654hiefjPPNm6/qdFHMCpeE6bZt8eihf5FshLKzO8uhS16b5L8aZO8IR55ySvwOgttvz645uo819ZY7bQCohKINAJVQtAGgEoo2AFRC0QaASszi7vFjwnTlymaXeCmlfO/g5jnjTyUzD937n5Ov/MU01sXgLUzyZqf4ddfFI7dsifMzzsiu2exMpx4PP3x8mN98c/Ydv9OztdCJI8J04cIPhPnExI3B2LhLfPny+IoTE9kZ448meX+40waASijaAFAJRRsAKqFoA0AlFG0AqMQs7h4/MEwfPDQ+a/ZjDwXZKaeEY8+9RZd4HXa0lf/wh0sb2UG3fysce813Tw/zsbFLp7UyZq+vf73ZJfzye+J98IEP7JfM8lwXV8TMvS5MJyaeScYf1EhOPDEeecMN1ydzPLjvZQ2AO20AqISiDQCVULQBoBKKNgBUYhY3or0lTCc3/pcw/5UgG7rlRd1bDgOQNQOtC9PXnNpsRCvnxsfefu5zWQNSdvgttRgfD8I3HJeMjo/BZLZZmeSfDtOzz76sjbnrei1wpw0AlVC0AaASijYAVELRBoBKKNoAUIlZ2z2+atXaML/5M60wv/DUjUH6iS6uiP47JkyXLFkdDz/ttEY0el6yBy7PPuCeesRHHa9b18zOeux/9XYp9FhwTnUppZSzw3Tx4mZ27bVRjSillCyfndxpA0AlFG0AqISiDQCVULQBoBIdN6K1Ws83hO3q0lJ+2XPPxUfIPfts9h1PB9lEt5bTA7sft6nHsT5Ta9/ZoytE51GWMjkZ742ndjbXMTaWHUUYz13KL6axrm7avea5sQ9681qQi5/f8f7IXjhmw2vEXHot6NXjmf38xsJ0585oD0Q1opT89aufe2P3taazB4ZaHe6ULVu2lBUrVnTyrbzAyMhIWb58+aCX0RF7oHvsA+wBprMHOi7ak5OTZXR0tAwPD5ehoaGOFjiftVqtMjY2VpYtW1YWLKjzfynsgZmzD7AHaGcPdFy0AYD+qvPXOgCYhxRtAKiEog0AlVC0AaASijYAVELRBoBKKNoAUAlFGwAqoWgDQCUUbQCohKINAJVQtAGgEh1/nrZPdZkZn+xDKfYB9gDt7YGOi/bo6KjPT+2Cmj9D1x7oHvsAe4Dp7IGOi/bw8PCeP723lLKo02nmsZ2llBtf8DjWZ2rt/6OUcuAgl1KxZ0opb50j++AvSykvHuRSKvVsKeWdc2QPnFxmUFbmsV2llFuntQc6fnSn/glkUVG0O1fzPyVNrf3AomjPzNzYBy8u9kHn5sYe2L+UsnCQS6nadPZAnf+BAgDzkKINAJVQtAGgEoo2AFRC0QaASijaAFAJRRsAKqFoA0AlFG0AqITz5oABWBxk+4UjFy5cG+YTE08lc98fZM9Ma1XU5ogkj/bG9l4upG/caQNAJRRtAKiEog0AlVC0AaASfW5Ee7i/l+u5qJnm8L6voi73Btnbk7Hv6+E6uuXfB9nSvq+iPlGj0OfCkRMTZ4T50NB7k7mbjWut1uPTXNfzngvTxYsPC/Px8Wj8HW1ek/YNoqa8MslfGmQbu351d9oAUAlFGwAqoWgDQCUUbQCohKINAJXoc/f4u4PsK/1dQleND3oBFWp2Al911SHhyB/84KYwX7MmnnnDhmb2eNI0vHFjfKzlKaccGOa33PLheKLk6E327qKL3tTIHnmkmZVSyne/G8+xdevWZPZ2jrA8KMnjn/f4eLzGUk5PcnKLgmxn31fRvgeTPH7HQbe50waASijaAFAJRRsAKqFoA0AlFG0AqESfu8ej3xGyrstVbc4ddYc+moz9n23OnTkyyCa6NPdcdXsjufji5cnY6Gz3Ur7xjbjbPH7ss+7gC8N0zZo/C/NbbrkvzBcu/GRzFRN3Jtfkeddc08yOPjoeu3XrA8ks2b6J3gEQdSqXUsr3kjx7XYrfdVDKzUF2VDKW3aJO8eg1tZT8eTyZ5PG7QGLfb2Ps3vy8S/PsnTttAKiEog0AlVC0AaASijYAVELRBoBK9Ll7/I42xv44ybOuwKjTOOv07JaXBFl2xjG7LQ2y7Mze7OeXdYzuCLJ7w5ErV8Zd4uvXb0zm/nKYvuxlzWxkJJmC/29s7PZGtjF76FPZue9bgizrNI/32OLFvxvm4+PZa1h/zp2e+7J3CmQOS/LonUPZ2G45PMge7vpV3GkDQCUUbQCohKINAJVQtAGgEn1uRGtH1tgRfcB9lnerCeA1Sd7rRjdiUcNZ5vowHR/Pjqn8RJiuXXtTmG/cuLWNtdBd2fM7albdlIw9JkzHx7PndnZE7dokp7eyo6pnOnZvjkjy/rwWuNMGgEoo2gBQCUUbACqhaANAJRRtAKjELO4eb1f3j4ubsijJH+/hNWnf/UH2yXDk1q1/HeZDQ3GX+K5d2TV7ue/Yu+wY03aOFD02yePjb3WJU8rONvPucqcNAJVQtAGgEoo2AFRC0QaASijaAFCJOdQ93g3HJ3l23jmDkXUHLw+yLcnYvwrTNWtODfO7775vn6ui36IzxkuJO/rfkIzNOtAfSvKle1sQc0r2sx5sPXCnDQCVULQBoBKKNgBUQtEGgEoo2gBQiQq7xwdx1vMzA7gmuS8k+eeD7Kxw5JIl8RnjO3Zk1/QOgsHJOryzn0nzXQQrVx4SjnzooS9New7mm2zfDbYeuNMGgEoo2gBQCUUbACqhaANAJRRtAKjELO4ez86M7oY1Sb6th9ekfdn5z5lbpz1yfDzON236UZvXpPeye4udYbpkydpG9tBD2c/1sM6WRIUWJXm2Bx7t1UJmxJ02AFRC0QaASijaAFAJRRsAKjGLG9Ge6+Hc23s4N91zUJhedNGfhfn69c1mo1e/Oj6u9O67tybXTM8xZWAOT/J4fzz22F1B+qlkjvM6WRBVympKVg/iRsdBc6cNAJVQtAGgEoo2AFRC0QaASijaAFCJWdw9/u4kvy3Jow7TxcnYrFswOduSAVkapuvXxx9Cv2TJsY1scbYFBvxB9kzfF75wZJgff3w8/thj9wvSk7q3ICqwJsieTsb+rIfr6D532gBQCUUbACqhaANAJRRtAKhEx41orVZrz596ddTbs0meXe8Xbcydje3nsXW7rzX1ONZnau29aurKfh5PhenkZPOYwl27srnHkrzfDWq7rzc39kH2nJ2ZHTvin/fTWV9R+LPNjqedDQ2Jux+3ubEH0idcn0Wv8dnr/kSbeS/sftymswc6LtpjY88/MW7sdIp9uLZH884uY2Nj5eCDDx70MjoytQfeOtB1PO/xx6eXzUZzYx+8syfzr1vXk2lnnbmxB24d6Dqm/M2gF9CR6eyBoVaHv95NTk6W0dHRMjw8XIaGhjpa4HzWarXK2NhYWbZsWVmwoM7/pbAHZs4+wB6gnT3QcdEGAPqrzl/rAGAeUrQBoBKKNgBUQtEGgEoo2gBQCUUbACqhaANAJRRtAKiEog0AlVC0AaASijYAVELRBoBKdPzRnD7VZWZ8sg+l2AfYA7S3Bzou2qOjo2XFihWdfjt7jIyMlOXLlw96GR2xB7rHPsAeYDp7oOOiPTw8vOdPnyulHNDpNPPYjlLKh17wONZnau3nlFIWDXIpFdtZSrlhjuyDk8sMXlLmsV2llFvtgXlt+nug40d36p9ADiilvLjTaea9mv8paWrti0opLxrkUqo3N/bB/qWUhYNcStXsAaazB+r8DxQAmIcUbQCohKINAJVQtAGgEoo2AFRC0QaASijaAFAJRRsAKuHoGvrssiDbmYw9KMm3J/m3guz+fa6I2eLIJN+S5OO9WgjMWu60AaASijYAVELRBoBKKNoAUIk+N6J9P8ge7u8S9mppkh+e5Mf0aiFz1p//+Usb2Q9/GI/91MeeDfOfbYs/b/bl9x7RDJccF0++a1ecb9sWxkNvfVOYv/71ixvZHXdcEM/NPjyQ5H+Q5M29tNtdQXZnm2vJXgtOT/JLgyxrmITOudMGgEoo2gBQCUUbACqhaANAJRRtAKhEX7vHV6z4fCMbGbk+Gf2jJH8wyaPu3uz4w/uSfGub+ZuD7GfJWEop5cwzo87q+Ge6fn02S3bc5SuDbHMy9i1hetVV/yoZ/+0wveOOo5Px7F30DoDs+fr7vVxI4t4kPz7J3xBk3+jOUuAF3GkDQCUUbQCohKINAJVQtAGgEoo2AFSir93j8XHP2QfZvz7J1yb5gW2MzbpUn0ry2NDQ6xpZq6V7fO92dmGO7IzqLI9E51OXcvHF2XnRUWd6KaV8K8gWtbGO+eq9QZY9/+LO/VLWJHm0D3YkY+9P8swn2hxPe34zyYPPFSillHJIkjffYbJixUHhyPe8J57hyis/ncy9Lsk/F2S3JmM7504bACqhaANAJRRtAKiEog0AlVC0AaASfe0e37r1vwdp1v2XOSzJo67f7yRjT07y7Lzzy8O01crGk4s6q7PO7Jlbu/azYb5x40TyHW9P8uy88+yMavbuii7Mkb1bYHGQZe9SYXbp3XntIyNxfuWV0Tn4pZSyOkyHh/cL87Gx9t591Cl32gBQCUUbACqhaANAJRRtAKhEXxvR2pM1Cj3exhwbkzw7Eu/BNq+ZjWcwmkffxkfnlpI3KWZH32bHEWZ7icHRdEY74mOtjz76T8N806b7knmyetNd7rQBoBKKNgBUQtEGgEoo2gBQCUUbACoxi7vH2xV1AGbHld6V5F9N8qxLXJfqYETHVJbyoQ+d0ci+9rVsjmuS/Pgkz446BOp2WZg+80w2/tM9W8l0uNMGgEoo2gBQCUUbACqhaANAJRRtAKjEHOoe/16QvTIZ2+6Z4T9qczy9tGLFJ8N8/2A3b92a/eyOTPL4HOJSFu1rWUCV4teCkZHs8wYe7d1SpsGdNgBUQtEGgEoo2gBQCUUbACqhaANAJSrsHs8Ok353kH0lGfv5JL+n7dXQS/HZ8SeeGI+++ur7g/T327xm9o4DoGannHJTmI+MxOM3bbq2h6vpnDttAKiEog0AlVC0AaASijYAVELRBoBKzOLu8Z8n+c42xq9Nxj6e5A/tdUX019FHnx7m27Zl33FXkJ2bjP1GByuiXq9L8u/3dRX0S/OdJ8cdF4+85ZbsjPFnurecLnKnDQCVULQBoBKKNgBUQtEGgErM4ka0/ZI8ay5rNhMsWRIfW/fYY7d3tCL66x//Mc6Hh7PviI6tzfbREe0viIppOJtPVq1a18iuuy4b/ZkerqT73GkDQCUUbQCohKINAJVQtAGgEoo2AFRi1naPn3vuh8L82mufSr7jgkby2GPjydj0HExmkYmJfxvmGzdmR9lGFnZnMQzQuiT/TB/XQE3Gg5f+xx77dv8X0gPutAGgEoo2AFRC0QaASijaAFCJjhvRWq3Wnj/t6NJSftnOnVnDWZZHzUnZ2GeTfGKva+qu3Y/b1ONYn6m1t9MY1o7s59HOzyk7xvQXba6lV3Y/dnNjH+zq0RWyzzXu5/O1l3Y/bvZA90xORq/9s+F1PzP9PdBx0R4bG9vzp7jLe6ZuvPH9XZglOot6dhkbGysHH3zwoJfRkak9cMNA1zEXzI190Dz/vzv+pkfzzi72QPf80z/V+zjuaw8MtTr89W5ycrKMjo6W4eHhMjQ01NEC57NWq1XGxsbKsmXLyoIFdf4vhT0wc/YB9gDt7IGOizYA0F91/loHAPOQog0AlVC0AaASijYAVELRBoBKKNoAUAlFGwAqoWgDQCUUbQCohKINAJVQtAGgEoo2AFTi/wFlgxm+j8P6JwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 16 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_conv_weights((outputs[1].permute(2,3,0,1).detach().cpu().numpy()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "mm = model.float()\n",
    "filters = mm.modules\n",
    "body_model = [i for i in mm.children()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method Module.modules of Net(\n",
       "  (conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\n",
       "  (conv2): Conv2d(16, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\n",
       "  (relu): ReLU()\n",
       "  (fc1): Linear(in_features=1764, out_features=128, bias=True)\n",
       "  (fc2): Linear(in_features=128, out_features=10, bias=True)\n",
       ")>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "filters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),\n",
       " Conv2d(16, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),\n",
       " ReLU(),\n",
       " Linear(in_features=1764, out_features=128, bias=True),\n",
       " Linear(in_features=128, out_features=10, bias=True)]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "body_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "layer1 = body_model[1]\n",
    "tensor = layer1.weight.permute(2,3,1,0).data.cpu().numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "36\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 36 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_conv_weights(tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/5], Step [100/4000], Loss: 2.2809\n",
      "Epoch [1/5], Step [200/4000], Loss: 2.2501\n",
      "Epoch [1/5], Step [300/4000], Loss: 2.0848\n",
      "Epoch [1/5], Step [400/4000], Loss: 1.4624\n",
      "Epoch [1/5], Step [500/4000], Loss: 0.5721\n",
      "Epoch [1/5], Step [600/4000], Loss: 0.4749\n",
      "Epoch [1/5], Step [700/4000], Loss: 0.1733\n",
      "Epoch [1/5], Step [800/4000], Loss: 0.2332\n",
      "Epoch [1/5], Step [900/4000], Loss: 0.2092\n",
      "Epoch [1/5], Step [1000/4000], Loss: 0.5266\n",
      "Epoch [1/5], Step [1100/4000], Loss: 0.3391\n",
      "Epoch [1/5], Step [1200/4000], Loss: 0.6980\n",
      "Epoch [1/5], Step [1300/4000], Loss: 0.1428\n",
      "Epoch [1/5], Step [1400/4000], Loss: 0.1686\n",
      "Epoch [1/5], Step [1500/4000], Loss: 0.1722\n",
      "Epoch [1/5], Step [1600/4000], Loss: 0.1498\n",
      "Epoch [1/5], Step [1700/4000], Loss: 0.3852\n",
      "Epoch [1/5], Step [1800/4000], Loss: 0.0663\n",
      "Epoch [1/5], Step [1900/4000], Loss: 0.1704\n",
      "Epoch [1/5], Step [2000/4000], Loss: 0.3970\n",
      "Epoch [1/5], Step [2100/4000], Loss: 0.2378\n",
      "Epoch [1/5], Step [2200/4000], Loss: 0.1341\n",
      "Epoch [1/5], Step [2300/4000], Loss: 0.1149\n",
      "Epoch [1/5], Step [2400/4000], Loss: 0.3566\n",
      "Epoch [1/5], Step [2500/4000], Loss: 0.5839\n",
      "Epoch [1/5], Step [2600/4000], Loss: 0.1255\n",
      "Epoch [1/5], Step [2700/4000], Loss: 0.0677\n",
      "Epoch [1/5], Step [2800/4000], Loss: 0.1436\n",
      "Epoch [1/5], Step [2900/4000], Loss: 0.0068\n",
      "Epoch [1/5], Step [3000/4000], Loss: 0.1557\n",
      "Epoch [1/5], Step [3100/4000], Loss: 0.0631\n",
      "Epoch [1/5], Step [3200/4000], Loss: 0.0120\n",
      "Epoch [1/5], Step [3300/4000], Loss: 0.0088\n",
      "Epoch [1/5], Step [3400/4000], Loss: 0.0471\n",
      "Epoch [1/5], Step [3500/4000], Loss: 0.1242\n",
      "Epoch [1/5], Step [3600/4000], Loss: 0.0925\n",
      "Epoch [1/5], Step [3700/4000], Loss: 0.0761\n",
      "Epoch [1/5], Step [3800/4000], Loss: 0.0111\n",
      "Epoch [1/5], Step [3900/4000], Loss: 0.0437\n",
      "Epoch [1/5], Step [4000/4000], Loss: 0.0253\n",
      "Epoch [2/5], Step [100/4000], Loss: 0.2742\n",
      "Epoch [2/5], Step [200/4000], Loss: 0.7490\n",
      "Epoch [2/5], Step [300/4000], Loss: 0.0604\n",
      "Epoch [2/5], Step [400/4000], Loss: 0.0958\n",
      "Epoch [2/5], Step [500/4000], Loss: 0.0287\n",
      "Epoch [2/5], Step [600/4000], Loss: 0.1749\n",
      "Epoch [2/5], Step [700/4000], Loss: 0.1532\n",
      "Epoch [2/5], Step [800/4000], Loss: 0.0913\n",
      "Epoch [2/5], Step [900/4000], Loss: 0.0327\n",
      "Epoch [2/5], Step [1000/4000], Loss: 0.1256\n",
      "Epoch [2/5], Step [1100/4000], Loss: 0.0347\n",
      "Epoch [2/5], Step [1200/4000], Loss: 0.0041\n",
      "Epoch [2/5], Step [1300/4000], Loss: 0.0610\n",
      "Epoch [2/5], Step [1400/4000], Loss: 0.0109\n",
      "Epoch [2/5], Step [1500/4000], Loss: 0.1554\n",
      "Epoch [2/5], Step [1600/4000], Loss: 0.0931\n",
      "Epoch [2/5], Step [1700/4000], Loss: 0.0132\n",
      "Epoch [2/5], Step [1800/4000], Loss: 0.2707\n",
      "Epoch [2/5], Step [1900/4000], Loss: 0.0050\n",
      "Epoch [2/5], Step [2000/4000], Loss: 0.0296\n",
      "Epoch [2/5], Step [2100/4000], Loss: 0.0016\n",
      "Epoch [2/5], Step [2200/4000], Loss: 0.4887\n",
      "Epoch [2/5], Step [2300/4000], Loss: 0.0870\n",
      "Epoch [2/5], Step [2400/4000], Loss: 0.0431\n",
      "Epoch [2/5], Step [2500/4000], Loss: 0.0113\n",
      "Epoch [2/5], Step [2600/4000], Loss: 0.0164\n",
      "Epoch [2/5], Step [2700/4000], Loss: 0.0171\n",
      "Epoch [2/5], Step [2800/4000], Loss: 0.0649\n",
      "Epoch [2/5], Step [2900/4000], Loss: 0.0619\n",
      "Epoch [2/5], Step [3000/4000], Loss: 0.0082\n",
      "Epoch [2/5], Step [3100/4000], Loss: 0.0082\n",
      "Epoch [2/5], Step [3200/4000], Loss: 0.0136\n",
      "Epoch [2/5], Step [3300/4000], Loss: 0.0392\n",
      "Epoch [2/5], Step [3400/4000], Loss: 0.0091\n",
      "Epoch [2/5], Step [3500/4000], Loss: 0.0049\n",
      "Epoch [2/5], Step [3600/4000], Loss: 0.0211\n",
      "Epoch [2/5], Step [3700/4000], Loss: 0.0109\n",
      "Epoch [2/5], Step [3800/4000], Loss: 0.0685\n",
      "Epoch [2/5], Step [3900/4000], Loss: 0.0072\n",
      "Epoch [2/5], Step [4000/4000], Loss: 0.0513\n",
      "Epoch [3/5], Step [100/4000], Loss: 0.0358\n",
      "Epoch [3/5], Step [200/4000], Loss: 0.0643\n",
      "Epoch [3/5], Step [300/4000], Loss: 0.0432\n",
      "Epoch [3/5], Step [400/4000], Loss: 0.0174\n",
      "Epoch [3/5], Step [500/4000], Loss: 0.0072\n",
      "Epoch [3/5], Step [600/4000], Loss: 0.0345\n",
      "Epoch [3/5], Step [700/4000], Loss: 0.0128\n",
      "Epoch [3/5], Step [800/4000], Loss: 0.0043\n",
      "Epoch [3/5], Step [900/4000], Loss: 0.0049\n",
      "Epoch [3/5], Step [1000/4000], Loss: 0.0439\n",
      "Epoch [3/5], Step [1100/4000], Loss: 0.0090\n",
      "Epoch [3/5], Step [1200/4000], Loss: 0.0184\n",
      "Epoch [3/5], Step [1300/4000], Loss: 0.0285\n",
      "Epoch [3/5], Step [1400/4000], Loss: 0.0020\n",
      "Epoch [3/5], Step [1500/4000], Loss: 0.6209\n",
      "Epoch [3/5], Step [1600/4000], Loss: 0.0168\n",
      "Epoch [3/5], Step [1700/4000], Loss: 0.0145\n",
      "Epoch [3/5], Step [1800/4000], Loss: 0.0043\n",
      "Epoch [3/5], Step [1900/4000], Loss: 0.0801\n",
      "Epoch [3/5], Step [2000/4000], Loss: 0.0075\n",
      "Epoch [3/5], Step [2100/4000], Loss: 0.0259\n",
      "Epoch [3/5], Step [2200/4000], Loss: 0.0779\n",
      "Epoch [3/5], Step [2300/4000], Loss: 0.0051\n",
      "Epoch [3/5], Step [2400/4000], Loss: 0.0291\n",
      "Epoch [3/5], Step [2500/4000], Loss: 0.0161\n",
      "Epoch [3/5], Step [2600/4000], Loss: 0.1350\n",
      "Epoch [3/5], Step [2700/4000], Loss: 0.0044\n",
      "Epoch [3/5], Step [2800/4000], Loss: 0.0484\n",
      "Epoch [3/5], Step [2900/4000], Loss: 0.0116\n",
      "Epoch [3/5], Step [3000/4000], Loss: 0.0121\n",
      "Epoch [3/5], Step [3100/4000], Loss: 0.0061\n",
      "Epoch [3/5], Step [3200/4000], Loss: 0.0111\n",
      "Epoch [3/5], Step [3300/4000], Loss: 0.3169\n",
      "Epoch [3/5], Step [3400/4000], Loss: 0.0382\n",
      "Epoch [3/5], Step [3500/4000], Loss: 0.1974\n",
      "Epoch [3/5], Step [3600/4000], Loss: 0.0010\n",
      "Epoch [3/5], Step [3700/4000], Loss: 0.0032\n",
      "Epoch [3/5], Step [3800/4000], Loss: 0.2896\n",
      "Epoch [3/5], Step [3900/4000], Loss: 0.1971\n",
      "Epoch [3/5], Step [4000/4000], Loss: 0.0194\n",
      "Epoch [4/5], Step [100/4000], Loss: 0.0268\n",
      "Epoch [4/5], Step [200/4000], Loss: 0.0109\n",
      "Epoch [4/5], Step [300/4000], Loss: 0.0182\n",
      "Epoch [4/5], Step [400/4000], Loss: 0.0100\n",
      "Epoch [4/5], Step [500/4000], Loss: 0.0010\n",
      "Epoch [4/5], Step [600/4000], Loss: 0.0069\n",
      "Epoch [4/5], Step [700/4000], Loss: 0.0145\n",
      "Epoch [4/5], Step [800/4000], Loss: 0.2898\n",
      "Epoch [4/5], Step [900/4000], Loss: 0.0103\n",
      "Epoch [4/5], Step [1000/4000], Loss: 0.0343\n",
      "Epoch [4/5], Step [1100/4000], Loss: 0.0130\n",
      "Epoch [4/5], Step [1200/4000], Loss: 0.0005\n",
      "Epoch [4/5], Step [1300/4000], Loss: 0.0030\n",
      "Epoch [4/5], Step [1400/4000], Loss: 0.3434\n",
      "Epoch [4/5], Step [1500/4000], Loss: 0.3012\n",
      "Epoch [4/5], Step [1600/4000], Loss: 0.0130\n",
      "Epoch [4/5], Step [1700/4000], Loss: 0.0058\n",
      "Epoch [4/5], Step [1800/4000], Loss: 0.0245\n",
      "Epoch [4/5], Step [1900/4000], Loss: 0.0310\n",
      "Epoch [4/5], Step [2000/4000], Loss: 0.0016\n",
      "Epoch [4/5], Step [2100/4000], Loss: 0.0050\n",
      "Epoch [4/5], Step [2200/4000], Loss: 0.2308\n",
      "Epoch [4/5], Step [2300/4000], Loss: 0.0019\n",
      "Epoch [4/5], Step [2400/4000], Loss: 0.1577\n",
      "Epoch [4/5], Step [2500/4000], Loss: 0.0780\n",
      "Epoch [4/5], Step [2600/4000], Loss: 0.1806\n",
      "Epoch [4/5], Step [2700/4000], Loss: 0.0142\n",
      "Epoch [4/5], Step [2800/4000], Loss: 0.0802\n",
      "Epoch [4/5], Step [2900/4000], Loss: 0.0082\n",
      "Epoch [4/5], Step [3000/4000], Loss: 0.0480\n",
      "Epoch [4/5], Step [3100/4000], Loss: 0.0211\n",
      "Epoch [4/5], Step [3200/4000], Loss: 0.0579\n",
      "Epoch [4/5], Step [3300/4000], Loss: 0.0019\n",
      "Epoch [4/5], Step [3400/4000], Loss: 0.0084\n",
      "Epoch [4/5], Step [3500/4000], Loss: 0.0180\n",
      "Epoch [4/5], Step [3600/4000], Loss: 0.0017\n",
      "Epoch [4/5], Step [3700/4000], Loss: 0.0042\n",
      "Epoch [4/5], Step [3800/4000], Loss: 0.0066\n",
      "Epoch [4/5], Step [3900/4000], Loss: 0.0023\n",
      "Epoch [4/5], Step [4000/4000], Loss: 0.0138\n",
      "Epoch [5/5], Step [100/4000], Loss: 0.0047\n",
      "Epoch [5/5], Step [200/4000], Loss: 0.0014\n",
      "Epoch [5/5], Step [300/4000], Loss: 0.0099\n",
      "Epoch [5/5], Step [400/4000], Loss: 0.0555\n",
      "Epoch [5/5], Step [500/4000], Loss: 0.0015\n",
      "Epoch [5/5], Step [600/4000], Loss: 0.0074\n",
      "Epoch [5/5], Step [700/4000], Loss: 0.0010\n",
      "Epoch [5/5], Step [800/4000], Loss: 0.0022\n",
      "Epoch [5/5], Step [900/4000], Loss: 0.0391\n",
      "Epoch [5/5], Step [1000/4000], Loss: 0.0024\n",
      "Epoch [5/5], Step [1100/4000], Loss: 0.0209\n",
      "Epoch [5/5], Step [1200/4000], Loss: 0.0714\n",
      "Epoch [5/5], Step [1300/4000], Loss: 0.0016\n",
      "Epoch [5/5], Step [1400/4000], Loss: 0.0636\n",
      "Epoch [5/5], Step [1500/4000], Loss: 0.0070\n",
      "Epoch [5/5], Step [1600/4000], Loss: 0.0132\n",
      "Epoch [5/5], Step [1700/4000], Loss: 0.0140\n",
      "Epoch [5/5], Step [1800/4000], Loss: 0.0040\n",
      "Epoch [5/5], Step [1900/4000], Loss: 0.0271\n",
      "Epoch [5/5], Step [2000/4000], Loss: 0.0532\n",
      "Epoch [5/5], Step [2100/4000], Loss: 0.0096\n",
      "Epoch [5/5], Step [2200/4000], Loss: 0.0004\n",
      "Epoch [5/5], Step [2300/4000], Loss: 0.0028\n",
      "Epoch [5/5], Step [2400/4000], Loss: 0.0083\n",
      "Epoch [5/5], Step [2500/4000], Loss: 0.0397\n",
      "Epoch [5/5], Step [2600/4000], Loss: 0.0582\n",
      "Epoch [5/5], Step [2700/4000], Loss: 0.0009\n",
      "Epoch [5/5], Step [2800/4000], Loss: 0.0049\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [5/5], Step [2900/4000], Loss: 0.0022\n",
      "Epoch [5/5], Step [3000/4000], Loss: 0.0053\n",
      "Epoch [5/5], Step [3100/4000], Loss: 0.0020\n",
      "Epoch [5/5], Step [3200/4000], Loss: 0.0002\n",
      "Epoch [5/5], Step [3300/4000], Loss: 0.0019\n",
      "Epoch [5/5], Step [3400/4000], Loss: 0.0582\n",
      "Epoch [5/5], Step [3500/4000], Loss: 0.0151\n",
      "Epoch [5/5], Step [3600/4000], Loss: 0.0053\n",
      "Epoch [5/5], Step [3700/4000], Loss: 0.0062\n",
      "Epoch [5/5], Step [3800/4000], Loss: 0.0044\n",
      "Epoch [5/5], Step [3900/4000], Loss: 0.0606\n",
      "Epoch [5/5], Step [4000/4000], Loss: 0.0503\n"
     ]
    }
   ],
   "source": [
    "model.train()\n",
    "\n",
    "total_step = len(train_loader)\n",
    "for epoch in range(num_epochs):\n",
    "    for i, (images, labels) in enumerate(train_loader):\n",
    "        images = images.to(device)\n",
    "        labels = labels.to(device)\n",
    "        \n",
    "        # Forward pass\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs[0], labels)\n",
    "        \n",
    "        # Backward and optimize\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        if (i+1) % 100 == 0:\n",
    "            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' \n",
    "                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the test-set into smaller batches of this size.\n",
    "test_batch_size = batch_size\n",
    "model.eval() \n",
    "def print_test_accuracy(show_example_errors=False,\n",
    "                        show_confusion_matrix=False):\n",
    "\n",
    "    num_test = test_dataset.data.cpu().numpy().shape[0]\n",
    "    cls_pred = np.zeros(shape=num_test, dtype=np.int)\n",
    "    cls_true = np.zeros(shape=num_test, dtype=np.int)\n",
    "    i = 0\n",
    "    with torch.no_grad():\n",
    "        correct = 0\n",
    "        total = 0\n",
    "        for images, labels in test_loader:\n",
    "            images = images.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = model(images)\n",
    "            _, predicted = torch.max(outputs[0].data, 1)\n",
    "            total += labels.size(0)\n",
    "            #pdb.set_trace()\n",
    "            correct += (predicted == labels).sum().item()\n",
    "            # The ending index for the next batch is denoted j.\n",
    "            j = min(i + test_batch_size, num_test)\n",
    "\n",
    "            # Calculate the predicted class using TensorFlow.\n",
    "            cls_pred[i:j] = predicted.detach().cpu().numpy()\n",
    "            cls_true[i:j] =labels.detach().cpu().numpy()\n",
    "            # Set the start-index for the next batch to the\n",
    "            # end-index of the current batch.\n",
    "            i = j\n",
    "\n",
    "    # Convenience variable for the true class-numbers of the test-set.\n",
    "\n",
    "\n",
    "    # Create a boolean array whether each image is correctly classified.\n",
    "    correct = (cls_true == cls_pred)\n",
    "\n",
    "    # Calculate the number of correctly classified images.\n",
    "    # When summing a boolean array, False means 0 and True means 1.\n",
    "    correct_sum = correct.sum()\n",
    "\n",
    "    # Classification accuracy is the number of correctly classified\n",
    "    # images divided by the total number of images in the test-set.\n",
    "    acc = float(correct_sum) / num_test\n",
    "\n",
    "    # Print the accuracy.\n",
    "    msg = \"Accuracy on Test-Set: {0:.1%} ({1} / {2})\"\n",
    "    print(msg.format(acc, correct_sum, num_test))\n",
    "\n",
    "    # Plot some examples of mis-classifications, if desired.\n",
    "    if show_example_errors:\n",
    "        print(\"Example errors:\")\n",
    "        plot_example_errors(cls_pred=cls_pred, correct=correct)\n",
    "\n",
    "    # Plot the confusion matrix, if desired.\n",
    "    if show_confusion_matrix:\n",
    "        print(\"Confusion Matrix:\")\n",
    "        plot_confusion_matrix(cls_pred=cls_pred, cls_true=cls_true)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_example_errors(cls_pred, correct):\n",
    "    incorrect = (correct == False)\n",
    "    images = test_dataset.data.cpu().numpy()[incorrect]\n",
    "    cls_pred = cls_pred[incorrect]\n",
    "    cls_true = test_dataset.targets.cpu().numpy()[incorrect]\n",
    "    plot_images(images=images[0:9],\n",
    "                cls_true=cls_true[0:9],\n",
    "                cls_pred=cls_pred[0:9])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000,)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_dataset.targets.cpu().numpy().shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_confusion_matrix(cls_pred,cls_true):\n",
    "    cm = confusion_matrix(y_true=cls_true,\n",
    "                          y_pred=cls_pred)\n",
    "    print(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ahmedkm\\AppData\\Local\\Temp\\ipykernel_11100\\1522424286.py:8: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  cls_pred = np.zeros(shape=num_test, dtype=np.int)\n",
      "C:\\Users\\ahmedkm\\AppData\\Local\\Temp\\ipykernel_11100\\1522424286.py:9: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  cls_true = np.zeros(shape=num_test, dtype=np.int)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on Test-Set: 98.8% (9876 / 10000)\n",
      "Example errors:\n"
     ]
    },
    {
     "data": {
      "image/png": "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BxYeHq61jY99NN62VuW5Nar+oqOjpfiRRx6ppJGoi3vMRERECmFhJiIiUggLMxERkUKqxByzfk7MOD+2aNEiKfb09NTa5Xm9pv62cvpH7gHF58H5WMia6/jx41Kszw1jLlsy92u8raH+sZAqnodBdEebNm2kuHXr1lJ84sQJs9d1P4+MVBn3mImIiBTCwkxERKSQKnEoW39IOi0tTepbvHixFAcGBmrtY8eOSX1Tp061egyzZ8+W4o8++khrGw8dTp8+3ertUPVivEWs/pCz8TImY97o8/7VV1+V+mJiYqRY/ySqiRMnSn28XI9UZsxlPz8/q9elf7qU8SlqVekyQe4xExERKYSFmYiISCEszERERAqxEcb795kpMzMTLi4uyMjIgLOzc1mPy2zGx+o9++yzWjsxMVHqs2SOYdSoUVJsnK++evWq1p47d67UV9mP1VPld1NVVNbPKycnR4qNOaa/zO7atWtS3+XLl6VYf57D77//LvVVpbk1I+ayZariz0v/XQoAkydP1trGRzcaFRYWSrF+jnn+/PlS3yuvvGLtEMuEJb8b7jETEREphIWZiIhIISzMRERECqkS1zGXxsfHR4qNj2A0l3GuWn+9KQBMmTJFioODg7V2VZ7Do8pTr149KTZeb6x/fOikSZOkvr///e9S3L17d63NfKSqxJivXbt21dr3mmOurrjHTEREpBAWZiIiIoVU+UPZRk8//bRVyxkPiWdlZZXFcIjKhPGJPHyCFFVX+mlCfbsm4R4zERGRQliYiYiIFMLCTEREpJBqN8dMVB0Z59pq6twbUU3APWYiIiKFsDATEREphIWZiIhIISzMRERECmFhJiIiUggLMxERkUJYmImIiBTCwkxERKQQFmYiIiKFWH3nLyEEACAzM7PMBkNl487v5M7viErHXFYXc9kyzGV1WZLLVhfmO49F9PT0tHYVVM6ysrLg4uJS2cNQHnNZfcxl8zCX1WdOLtsIK/8ULSoqwsWLF2EymYo9G5YqlxACWVlZ8PDwQK1anK24F+ayupjLlmEuq8uSXLa6MBMREVHZ45+gRERECmFhJiIiUggLMxERkUJYmImIiBTCwkxERKQQFmYiIiKFsDATEREphIWZiIhIISzMipg5cyYef/zxyh4G0X0bM2YMBg0aVNnDILpvlfW9bHVhtrGxKfXfzJkzy3CYlktISEDP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ERGRQliYiYiIFFIlDmXrDznrb38IAOvWrZPi6OhorX3o0KHyHRjVCKGhoVI8atQoKXZyctLaDz30kNnrvXXrlhRv27ZNivWXYRkP6ZVm//79Urxx40Ypfumll8xeF1FZSE5O1toDBgyQ+oyXI+oPTz/55JNSn3G6x3h7XP13Pg9lExERUZlgYSYiIlIICzMREZFClJxjNs69jRw5Umtv2LChzLbTv39/rW1vby/1rV69usy2Q1Wbu7t7qbG1jI9n1OcjADzzzDNa+4UXXpD6jPPGpXnllVekWP+ISAB49tlnzV4XkTmuX78uxSNGjNDaqampUp/+8igAaNeundYeNGiQ1HevSwrXrFmjtavyuRTcYyYiIlIICzMREZFCWJiJiIgUouQcc25urhSX1byy8Zq45cuXa23jbeE4x0yVTT8H/eOPP0p9zz33nBRv3ry5xPXk5ORIcb9+/aRYf4tER0dHi8dJdOLECSkeP368FOuvL37sscekPuPjePWef/55KTbebtZ4i84LFy7ce7BVAPeYiYiIFMLCTEREpBAlD2WXRn/pFADExMSU+F7jrd/0p+wDQP369bX2119/ff+DI6ogS5culeIHH3zQ6nXpL9P6+eefrV4P1Sz79u3T2hMmTJD6EhISpNjGxkZrDx48WOqztS25DLVo0aLUMejXC8i3/jQeXm/dunWp61IJ95iJiIgUwsJMRESkEBZmIiIihVSJOWZPT0+tfe7cOanP+MgwPZPJJMWlXQqyadMmK0dHVPEaNGggxT179tTaxkv/7kX/mMiUlBSpryo/Oo/K19y5c7W2cU65NLNmzZLiuLg4Kdbfftb4iNV7yc7O1trGWsE5ZiIiIrIKCzMREZFCWJiJiIgUouQcc7169aT422+/1dq3b9+W+po0aWL1dqKiorR2YWGh1eshqmjGx5T26NFDa1s6x1yr1v/9fV63bt37GxjVGPpzE+6lZcuWWrtr165S340bN6RYf010ZGSkdYOr4rjHTEREpBAWZiIiIoUoeSjbeJjOz8+vXLajv73nuHHjpL6CgoJSl9UfQg8PDy/bgRFZ6K233tLa169fl/o+/fTTUpfNyMjQ2qNHj5b6du3aVQajo+po586dWts4fWK8zK5jx45mr1d/iLxv376lvtf4dKnqgnvMRERECmFhJiIiUggLMxERkUKUnGMuL4cPH5bi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\n",
      "text/plain": [
       "<Figure size 640x480 with 9 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[ 975    0    1    0    0    0    0    1    3    0]\n",
      " [   0 1125    2    1    0    1    1    1    4    0]\n",
      " [   1    1 1021    3    0    0    0    3    3    0]\n",
      " [   1    0    0 1002    0    2    0    2    2    1]\n",
      " [   0    0    3    0  964    0    1    3    1   10]\n",
      " [   2    0    0   10    0  878    2    0    0    0]\n",
      " [   5    2    0    1    2    4  943    0    1    0]\n",
      " [   0    2    5    1    0    0    0 1017    1    2]\n",
      " [   1    0    2    3    0    1    0    2  964    1]\n",
      " [   2    2    0    4    6    2    0    4    2  987]]\n"
     ]
    }
   ],
   "source": [
    "print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# Save the model checkpoint\n",
    "torch.save(model.state_dict(), 'model.ckpt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
